#![allow(dead_code)]

extern crate nn;
extern crate rand;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use self::nn::{NN, HaltCondition};
use self::rand::Rng;
use super::Player;
use super::super::field::Field;

const GAMMA:f64 = 0.99; //q gamma (action-reward time difference high) //1.0?
const LR:f64 = 0.2; //neural net learning rate
const MOM:f64 = 0.1; //neural net momentum
const EPOCHS_PER_STEP:u32 = 1; //epochs to learn from each turn
const RND_PICK_START:f64 = 0.5; //exploration factor start
const RND_PICK_DEC:f64 = 500000.0; //random exploration decrease factor^-1


pub struct PlayerAIQ
{
	initialized: bool,
	pid: i32, //player ID
	fixed: bool, //should the agent learn or not (fixed => dont learn)
	filename: String,
	games_played: u32,
	nn: Option<NN>,
}

impl PlayerAIQ
{
	pub fn new(fix:bool) -> Box<PlayerAIQ>
	{
		Box::new(PlayerAIQ { initialized: false, pid: 0, fixed: fix, filename: String::new(), games_played: 0, nn: None })
	}
	
	fn get_exploration(games_played:u32) -> f64
	{
		RND_PICK_START * (-(games_played as f64)/RND_PICK_DEC).exp()
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
	
	fn field_to_input(field:&Vec<i32>, p:i32) -> Vec<f64>
	{
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity(field.len() * 3);
		for val in field.iter()
		{ //3 nodes for every field
			input.push(if *val == 0 { 1f64 } else { 0f64}); //one for empty fields
			input.push(if *val == p { 1f64 } else { 0f64}); //one for self players nodes
			input.push(if *val == op { 1f64 } else { 0f64}); //one for other players nodes
		}
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
			//create new neural net, is it could not be loaded
			let n = field.get_size();
			self.nn = Some(NN::new(&[3*n, 4*n, 4*n, 4*n, 2*n, 2*n, 2*n, 2*n, n, n, n, n, n/2, n/2, n/2, n/2, n/4, n/4, n/4, n/4, field.get_w()])); //set size of NN layers here
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
			let state = PlayerAIQ::field_to_input(field.get_field(), self.pid);
			
			//choose action by e-greedy
			let mut qval = nn.run(&state);
			let mut x = PlayerAIQ::argmax(&qval);
			if rng.gen::<f64>() < PlayerAIQ::get_exploration(self.games_played) //random exploration
			{
				x = rng.gen::<u32>() % field.get_w();
			}
			
			//perform action and get reward
			#[allow(unused_assignments)]
			let mut reward:f64 = 0.0;
			res = field.play(self.pid, x);
			let flag = field.get_state();
			if res
			{
				if flag == -1 || flag == 0 { reward = 0.0; } //running game or draw
				else if flag == self.pid { reward = 1.0; } //win
				else { reward = -1.0; } //lose
			}
			else { reward = -1.0; } //move did not meet the rules
			
			//calculate NN update if not fixed, but learn if move did not was rule-conform
			if !self.fixed || !res
			{
				//get Q values for next state
				let state2 = PlayerAIQ::field_to_input(field.get_field(), self.pid);
				let qval2 = nn.run(&state2);
				let x2 = PlayerAIQ::argmax(&qval2);
				qval[x as usize] = reward + GAMMA * qval2[x2 as usize]; //Q learning (see https://www.reddit.com/r/MachineLearning/comments/1kc8o7/understanding_qlearning_in_neural_networks/)
				let training = [(state, qval)];
				nn.train(&training)
					.halt_condition(HaltCondition::Epochs(EPOCHS_PER_STEP))
					.log_interval(None)
					.momentum(MOM)
					.rate(LR)
					.go();
			}
		}
		
		res
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		self.games_played += 1;
	}
}

impl Drop for PlayerAIQ
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
