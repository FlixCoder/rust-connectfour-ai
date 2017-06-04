//! offline reinforcement learning (q learning after match is over)
#![allow(dead_code)]

extern crate rand;
extern crate nn;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use self::rand::Rng;
use self::nn::{NN, HaltCondition};
use super::Player;
use super::super::field::Field;

const GAMMA:f64 = 0.99; //temporal sureness (->1 means more sure about early actions always lead to win)
const LR:f64 = 0.1; //neural net learning rate
const LR_DECAY:f64 = 10000f64; //NN learning rate decrease (half every DECAY games)
const LR_MIN:f64 = 0.01; //minimum NN LR
const MOM:f64 = 0.05; //neural net momentum
const EPOCHS_PER_STEP:u32 = 1; //epochs to learn from each turn
const RND_PICK_START:f64 = 1.0f64; //exploration factor start
const RND_PICK_DEC:f64 = 20000f64; //random exploration decrease (half every DEC games)


pub struct PlayerAIQOff
{
	initialized: bool,
	fixed: bool, //should the agent learn or not (fixed => dont learn)
	filename: String,
	pid: i32, //player ID
	nn: Option<NN>, //neural network
	games_played: u32,
	lr: f64,
	exploration: f64,
	play_buffer: Vec<(Vec<f64>, Vec<f64>)>,
}

impl PlayerAIQOff
{
	pub fn new(fix:bool) -> Box<PlayerAIQOff>
	{
		Box::new(PlayerAIQOff { initialized: false, fixed: fix, filename: String::new(), pid: 0,
				nn: None, games_played: 0, lr: LR, exploration: RND_PICK_START,
				play_buffer: Vec::new() })
	}
	
	fn get_exploration(&self) -> f64
	{
		RND_PICK_START * (2f64).powf(-(self.games_played as f64)/RND_PICK_DEC)
	}
	
	fn get_lr(&self) -> f64
	{
		LR_MIN.max(LR * (2f64).powf(-(self.games_played as f64)/LR_DECAY))
	}
	
	fn argmax(slice:&[f64]) -> u32
	{
		let mut x:u32 = 0;
		let mut max = slice[0];
		for i in 1..slice.len()
		{
			if max<slice[i]
			{
				x = i as u32;
				max = slice[i];
			}
		}
		x
	}
	
	fn field_to_input(field:&mut Field, p:i32) -> Vec<f64>
	{
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity((2*field.get_size() + field.get_w()) as usize);
		for (i, val) in field.get_field().iter().enumerate()
		{ //2 nodes for every square: -1 enemy, 0 free, 1 own; 0 square will not be reached with one move, 1 square can be directly filled
			if *val == p { input.push(1f64); input.push(0f64); }
			else if *val == op { input.push(-1f64); input.push(0f64); }
			else
			{ //empty square
				input.push(0f64);
				if (i as u32) < (field.get_size()-field.get_w()) { input.push(if field.get_field()[i+field.get_w() as usize] != 0 { 1f64 } else { 0f64 }); }
				else { input.push(1f64); }
			}
		}
		for x in 0..field.get_w()
		{ //1 node for every column: 1 own win, -1 enemy win, 0 none (which consistent order of the nodes does not matter, fully connected)
			if field.play(p, x)
			{ //valid play
				match field.get_state()
				{
					-1 | 0 => input.push(0f64),
					pid => input.push(if pid == p {1f64} else {-1f64}),
				}
				field.undo();
			}
			else { input.push(0f64); } //illegal move, nobody can win
		}
		input
	}
}

impl Player for PlayerAIQOff
{
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		self.pid = p;
		
		self.filename = format!("AIQOff-{}x{}.NN", field.get_w(), field.get_h());
		let file = File::open(&self.filename);
		if file.is_err()
		{
			//create new neural net, is it could not be loaded
			let n = field.get_size();
			let w = field.get_w();
			self.nn = Some(NN::new(&[2*n+w, 4*n, 2*n, n, n, n/2, w])); //set size of NN layers here
			//games_played, exploration, lr already set
		}
		else
		{
			//load neural net from file (and games played)
			let mut reader = BufReader::new(file.unwrap());
			let mut datas = String::new();
			let mut nns = String::new();
			
			let res1 = reader.read_line(&mut datas);
			let res2 = reader.read_to_string(&mut nns);
			if res1.is_err() || res2.is_err() { return false; }
			
			let res = datas.trim().parse::<u32>();
			if res.is_err() { return false; }
			self.games_played = res.unwrap();
			self.nn = Some(NN::from_json(&nns));
			
			self.lr = self.get_lr();
			self.exploration = self.get_exploration();
		}
		
		self.initialized = true;
		true
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		//variables
		let mut rng = rand::thread_rng();
		let nn = self.nn.as_mut().unwrap();
		let mut res = false;
		
		//choose an action (try again until it meets the rules)
		while !res
		{
			//get current state formatted for the neural net (in loop because ownerships gets moved later)
			let state = PlayerAIQOff::field_to_input(field, self.pid);
			
			//choose action by e-greedy
			let mut qval = nn.run(&state);
			let mut x = PlayerAIQOff::argmax(&qval);
			if rng.gen::<f64>() < self.exploration //random exploration
			{
				x = rng.gen::<u32>() % field.get_w();
			}
			
			//perform action
			res = field.play(self.pid, x);
			
			//save play data if not fixed, but learn if move did not was rule-conform
			if !self.fixed || !res
			{
				//calculate q update or collect data for later learn
				if !res { qval[x as usize] = 0f64; } //invalid play instant learn
				else { qval[x as usize] = -1f64; } //mark field to set values later for learning
				
				//initiate training or save data
				if res
				{
					//add latest experience to replay_buffer
					self.play_buffer.push((state, qval));
				}
				else
				{
					let training = [(state, qval)];
					nn.train(&training)
						.halt_condition(HaltCondition::Epochs(EPOCHS_PER_STEP))
						.log_interval(None)
						//.log_interval(Some(2)) //debug
						.momentum(MOM)
						.rate(self.lr)
						.go();
				}
			}
		}
		//field.print(); //debug
		res
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		if self.initialized && !self.fixed
		{
			//get reward
			let otherp = if self.pid == 1 {2} else {1};
			let mut reward = 0f64; //draw (or running, should not happen)
			if state == self.pid { reward = 1f64; }
			else if state == otherp { reward = -1f64; }
			
			//learn
			let w:usize = field.get_w() as usize;
			let len = self.play_buffer.len() as i32;
			for count in 0..len
			{
				let mut qval = &mut self.play_buffer[count as usize].1;
				for i in 0..w
				{
					if qval[i] == -1f64
					{
						qval[i] = (GAMMA.powi(len - count) * reward + 1f64) / 2f64; // (+1)/2 => accumulate for sigmoid
						break;
					}
				}
			}
			
			let nn = self.nn.as_mut().unwrap();
			nn.train(&self.play_buffer)
						.halt_condition(HaltCondition::Epochs(EPOCHS_PER_STEP))
						.log_interval(None)
						//.log_interval(Some(2)) //debug
						.momentum(MOM)
						.rate(self.lr)
						.go();
		}
		//set parameters
		self.games_played += 1;
		self.lr = self.get_lr();
		self.exploration = self.get_exploration();
	}
}

impl Drop for PlayerAIQOff
{
	fn drop(&mut self)
	{
		//write neural net to file, if it may has learned and was initialized
		if self.initialized && !self.fixed
		{
			let file = File::create(&self.filename);
			if file.is_err() { println!("Warning: Could not write AIQ NN file!"); return; }
			let mut writer = BufWriter::new(file.unwrap());
			
			let res1 = writeln!(&mut writer, "{}", self.games_played);
			let res2 = write!(&mut writer, "{}", self.nn.as_mut().unwrap().to_json());
			if res1.is_err() || res2.is_err() { println!("Warning: There was an error while writing AIQ NN file!"); return; }
		}
	}
}
