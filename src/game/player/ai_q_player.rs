//! online reinforcement q learner (kind of double q)
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

const GAMMA:f64 = 0.99; //q gamma (action-reward time difference high) //1.0?
const LR:f64 = 0.5; //neural net learning rate
const LR_DECAY:f64 = 100000f64; //NN learning rate decrease (half every DECAY games)
const LR_MIN:f64 = 0.01; //minimum NN LR
const MOM:f64 = 0.1; //neural net momentum
const EPOCHS_PER_STEP:u32 = 1; //epochs to learn from each turn
const RND_PICK_START:f64 = 0.5; //exploration factor start
const RND_PICK_DEC:f64 = 100000f64; //random exploration decrease (half every DEC games)
const RND_PICK_MIN:f64 = 0.01; //exploration rate minimum


pub struct PlayerAIQ
{
	initialized: bool,
	fixed: bool, //should the agent learn or not (fixed => dont learn)
	filename: String,
	pid: i32, //player ID
	nn: Option<NN>, //online network
	targetnn: Option<NN>, //target network (temporarely fixed value network)
	games_played: u32,
	lr: f64,
	exploration: f64,
	startp: f64, //for NN input (1 = self starting, -1 = enemy starting)
	memstate: Vec<f64>, //memorize state learning next turn
	memqval: Vec<f64>, //same
	memreward: f64, //same
	memplay: u32, //same
}

impl PlayerAIQ
{
	pub fn new(fix:bool) -> Box<PlayerAIQ>
	{
		Box::new(PlayerAIQ { initialized: false, fixed: fix, filename: String::new(), pid: 0,
				nn: None, targetnn: None, games_played: 0, lr: LR, exploration: RND_PICK_START,
				startp: 0.0, memstate: Vec::new(), memqval: Vec::new(), memreward: -1.0, memplay: 0 })
	}
	
	fn get_exploration(&self) -> f64
	{
		RND_PICK_MIN.max(RND_PICK_START * (2f64).powf(-(self.games_played as f64)/RND_PICK_DEC))
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
	
	fn field_to_input(field:&mut Field, p:i32, startp:f64) -> Vec<f64>
	{
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity((2*field.get_size() + field.get_w() + 1) as usize);
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
		//1 node for starting player (-1 enemy, 1 self)
		input.push(startp);
		//return
		input
	}
}

impl Player for PlayerAIQ
{
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		self.pid = p;
		
		self.filename = format!("AIQ-{}x{}.NN", field.get_w(), field.get_h());
		let file = File::open(&self.filename);
		if file.is_err()
		{
			//create new neural net, as it could not be loaded
			let n = field.get_size();
			let w = field.get_w();
			self.nn = Some(NN::new(&[2*n+w+1, 4*n, 2*n, n, n, n/2, w])); //set size of NN layers here
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
		
		self.targetnn = self.nn.clone();
		self.initialized = true;
		true
	}
	
	fn startp(&mut self, p:i32)
	{
		if p == self.pid
		{
			self.startp = 1.0;
		}
		else
		{
			self.startp = -1.0;
		}
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		//variables
		let mut rng = rand::thread_rng();
		let nn = self.nn.as_mut().unwrap();
		let targetnn = self.targetnn.as_mut().unwrap();
		
		//get current state formatted for the neural net
		let state = PlayerAIQ::field_to_input(field, self.pid, self.startp);
		
		//learn if not first move (reward is already set, won/loose would be outcome)
		if self.memreward != -1.0
		{
			//get Q values for next state
			let qval2 = targetnn.run(&state); //use double q learning target nn, to decouple action and value a bit
			let max = qval2[PlayerAIQ::argmax(&qval2) as usize];
			//calculate q update
			self.memqval[self.memplay as usize] = (self.memreward + GAMMA * max) / (1.0 + GAMMA); //Q learning (divide to stay in [0,1] for sigmoid)
			//train on the latest experience (q update)
			nn.train(&[(self.memstate.clone(), self.memqval.clone())])
				.halt_condition(HaltCondition::Epochs(EPOCHS_PER_STEP))
				.log_interval(None)
				//.log_interval(Some(2)) //debug
				.momentum(MOM)
				.rate(self.lr)
				.go();
		}
		
		//choose action by e-greedy (no exploration when fixed AI version+)
		self.memqval = nn.run(&state);
		self.memplay = PlayerAIQ::argmax(&self.memqval);
		if !self.fixed && rng.gen::<f64>() < self.exploration //random exploration if agent should learn.
		{
			self.memplay = rng.gen::<u32>() % field.get_w();
		}
		
		//perform action and set reward
		self.memreward = 0.5;
		let mut res = field.play(self.pid, self.memplay);
		if !res { self.memqval[self.memplay as usize] = 0.1; } //move did not meet the rules, but learn anyway, even if another random move is made
		
		//random play when it was not rule conform, also modify q-value for it
		while !res //infinite if it is already draw! (cannot happen without bug)
		{
			self.memplay = rng.gen::<u32>() % field.get_w();
			res = field.play(self.pid, self.memplay);
		}
		
		//save state for next turn
		self.memstate = state;
		
		//return
		res
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		{ //learn, scope for "let nn" shortcut
		let nn = self.nn.as_mut().unwrap();
		let op:i32 = if self.pid == 1 { 2 } else { 1 }; //other player
		//set reward (if draw, reward already set properly)
		if state == self.pid { self.memreward = 1.0; }
		else if state == op { self.memreward = 0.0; }
		//end-values of network should meet reward exactly
		self.memqval[self.memplay as usize] = self.memreward;
		//train NN
		nn.train(&[(self.memstate.clone(), self.memqval.clone())])
			.halt_condition(HaltCondition::Epochs(EPOCHS_PER_STEP))
			.log_interval(None)
			//.log_interval(Some(2)) //debug
			.momentum(MOM)
			.rate(self.lr)
			.go();
		}
		
		//parameters
		self.games_played += 1;
		self.lr = self.get_lr();
		self.exploration = self.get_exploration();
		self.targetnn = self.nn.clone();
		self.memreward = -1.0;
	}
}

impl Drop for PlayerAIQ
{
	fn drop(&mut self)
	{
		//write neural net to file, if it was allowed to learn and was initialized
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
