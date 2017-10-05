//! online reinforcement q learner (kind of double q) with experience replay
#![allow(dead_code)]

extern crate rand;
extern crate nn;
extern crate rustc_serialize;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use self::rustc_serialize::json;
use self::rand::Rng;
use self::nn::{NN, HaltCondition, Activation};
use super::Player;
use super::super::field::Field;

const GAMMA:f64 = 0.99; //q gamma (action-reward time difference high) (not 1.0, it terminates)
const LR:f64 = 0.01; //neural net learning rate (deterministic -> high)
const LR_DECAY:f64 = 0.01 / 100000f64; //NN learning rate decrease per game(s)
const LR_MIN:f64 = 0.0001; //minimum NN LR
const LAMBDA:f64 = 0.0001; //L2 regularization parameter lambda (divide by n manually, pick very small > 0, like pick LAMBDA / n)
const MOM:f64 = 0.1; //neural net momentum
const RND_PICK_START:f64 = 0.5; //exploration factor start
const RND_PICK_DEC:f64 = 20000f64; //random exploration decrease (half every DEC games)
const RND_PICK_MIN:f64 = 0.05; //exploration rate minimum
const EXP_REP_SIZE:usize = 20000; //size of buffer for experience replay
const EXP_REP_BATCH:u32 = 19; //batch size for replay training
const EPOCHS:u32 = 1; //NN training epochs for a mini batch


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
	explore: bool, //should the agent explore randomly sometimes?
	startp: f64, //for NN input (1 = self starting, -1 = enemy starting)
	exp_buffer: Option<Vec<(Vec<f64>, usize, f64, Vec<f64>)>>, //experience buffer for experience replay
	memstate: Vec<f64>, //memorize state learning next turn
	memqval: Vec<f64>, //same
	memreward: f64, //same
	memplay: u32, //same
}

//reward values, take care of q-updates when changing (/(x + GAMMA))
const REW_WIN:f64 = 1.0;
const REW_LOSE:f64 = 0.0;
const REW_NORMAL:f64 = 0.5; //draw or just a middle play
const REW_WRONG:f64 = 0.1; //play against rules -> random pick
const REW_FLAG:f64 = -1000.0; //used as flag to indicate there was no reward yet. (so no play done)

impl PlayerAIQ
{
	pub fn new(fix:bool, exp:bool) -> Box<PlayerAIQ>
	{
		Box::new(PlayerAIQ { initialized: false, fixed: fix, filename: String::new(), pid: 0,
				nn: None, targetnn: None, games_played: 0, lr: LR, exploration: RND_PICK_START,
				explore: exp, startp: 0.0, exp_buffer: None,
				memstate: Vec::new(), memqval: Vec::new(), memreward: REW_FLAG, memplay: 0 })
	}
	
	fn get_exploration(&self) -> f64
	{
		RND_PICK_MIN.max(RND_PICK_START * (2f64).powf(-(self.games_played as f64)/RND_PICK_DEC))
	}
	
	fn get_lr(&self) -> f64
	{
		LR_MIN.max(LR - LR_DECAY * self.games_played as f64)
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
	
	//field and extra info
	fn field_to_input(field:&mut Field, p:i32, startp:f64) -> Vec<f64>
	{
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity((2*field.get_size() + field.get_w() + 1) as usize);
		//2 nodes for every square: -1 enemy, 0 free, 1 own; 0 square will not be reached with one move, 1 square can be directly filled
		for (i, val) in field.get_field().iter().enumerate()
		{
			if *val == p { input.push(1f64); input.push(0f64); }
			else if *val == op { input.push(-1f64); input.push(0f64); }
			else
			{ //empty square
				input.push(0f64);
				if (i as u32) < (field.get_size()-field.get_w()) { input.push(if field.get_field()[i+field.get_w() as usize] != 0 { 1f64 } else { 0f64 }); }
				else { input.push(1f64); }
			}
		}
		//1 node for every column: 1 a player can win, 0 none (which consistent order of the nodes does not matter, fully connected)
		for x in 0..field.get_w()
		{
			if field.play(p, x)
			{ //valid play
				match field.get_state()
				{ //if game was over before play (does not happen without bug), it looks like we win, even if we don't
					-1 | 0 =>
						{
							field.undo();
							field.play(op, x);
							match field.get_state()
							{
								-1 | 0 => input.push(0f64), //nobody can win
								_ => input.push(1f64), //enemy can win
							}
						},
					_ => input.push(1f64), //we can win
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
	
	//raw field
	/*fn field_to_input(field:&mut Field, p:i32, startp:f64) -> Vec<f64>
	{
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity((field.get_size() + 1) as usize);
		//1 nodes for every square: -1 enemy, 0 free, 1 own
		for val in field.get_field().iter()
		{
			if *val == p { input.push(1f64); }
			else if *val == op { input.push(-1f64); }
			else { input.push(0f64); } //empty square
		}
		//1 node for starting player (-1 enemy, 1 self)
		input.push(startp);
		//return
		input
	}*/
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
			self.nn = Some(NN::new(&[2*n+w+1, 3*n, n, w], Activation::PELU, Activation::Sigmoid)); //set size of NN layers here
			self.exp_buffer = Some(Vec::with_capacity(EXP_REP_SIZE));
			//games_played, exploration, lr already set
		}
		else
		{
			//load neural net from file (and games played)
			let mut reader = BufReader::new(file.unwrap());
			let mut datas = String::new();
			let mut nns = String::new();
			let mut exps = String::new();
			
			let res1 = reader.read_line(&mut datas);
			let res2 = reader.read_line(&mut nns);
			let res3 = reader.read_to_string(&mut exps);
			if res1.is_err() || res2.is_err() || res3.is_err() { return false; }
			
			let res = datas.trim().parse::<u32>();
			if res.is_err() { return false; }
			self.games_played = res.unwrap();
			self.nn = Some(NN::from_json(&nns));
			self.exp_buffer = Some(json::decode(&exps).unwrap());
			
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
		let mut exp_buffer = self.exp_buffer.as_mut().unwrap();
		
		//get current state formatted for the neural net
		let state = PlayerAIQ::field_to_input(field, self.pid, self.startp);
		
		//learn if not fixed and not first move (reward is already set, won/loose would be outcome)
		if !self.fixed && self.memreward != REW_FLAG
		{
			//get Q values for next state
			let qval2 = targetnn.run(&state); //use double q learning target nn, to decouple action and value a bit
			let max = qval2[PlayerAIQ::argmax(&qval2) as usize];
			//calculate q update
			self.memqval[self.memplay as usize] = (self.memreward + GAMMA * max) / (1.0 + GAMMA); //Q learning (divide to stay in [-1,1])
			//train on experience replay and the latest experience (q update)
			let mut trainingset = Vec::new();
			//experience
			if exp_buffer.len() > 0
			{
				for _ in 0..EXP_REP_BATCH
				{ //EXP_REP_BATCH random experiences to replay
					let repindex = rng.gen::<usize>() % exp_buffer.len();
					let mut qval = nn.run(&exp_buffer[repindex].0); //.0 = state 1
					if exp_buffer[repindex].2 == REW_LOSE || exp_buffer[repindex].2 == REW_WIN //.2 = reward
					{
						qval[exp_buffer[repindex].1] = exp_buffer[repindex].2; //.1 = action, .2 = reward
					}
					else
					{
						let qval2 = targetnn.run(&exp_buffer[repindex].3); //.3 = state 2
						let max = qval2[PlayerAIQ::argmax(&qval2) as usize];
						qval[exp_buffer[repindex].1] = (exp_buffer[repindex].2 + GAMMA * max) / (1.0 + GAMMA); //.1 = action, .2 = reward
					}
					trainingset.push((exp_buffer[repindex].0.clone(), qval));
				}
			}
			//latest
			trainingset.push((self.memstate.clone(), self.memqval.clone()));
			nn.train(&trainingset)
				.halt_condition(HaltCondition::Epochs(EPOCHS))
				.log_interval(None)
				//.log_interval(Some(2)) //debug
				.momentum(MOM)
				.rate(self.lr)
				.lambda(LAMBDA / (self.games_played as f64 + 1.0))
				.go();
			//save latest as experience
			if exp_buffer.len() >= EXP_REP_SIZE
			{
				exp_buffer.remove(0); //remove first element
			}
			let (state1, _) = trainingset.pop().unwrap();
			exp_buffer.push((state1, self.memplay as usize, self.memreward, state.clone())); //state1 = memstate (so state to choose action), state = current state (so next state)
		}
		
		//choose action by e-greedy
		self.memqval = nn.run(&state);
		self.memplay = PlayerAIQ::argmax(&self.memqval);
		if self.explore && rng.gen::<f64>() < self.exploration //random exploration if it should
		{
			self.memplay = rng.gen::<u32>() % field.get_w();
		}
		
		//perform action and set reward
		self.memreward = REW_NORMAL;
		let mut res = field.play(self.pid, self.memplay);
		if !res { self.memqval[self.memplay as usize] = REW_WRONG; } //move did not meet the rules, but learn anyway, even if another random move is made
		
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
		if !self.fixed
		{ //learn if not fixed (scope needed for "let nn" and "let targetnn" shortcut)
			let nn = self.nn.as_mut().unwrap();
			let targetnn = self.targetnn.as_mut().unwrap();
			let mut exp_buffer = self.exp_buffer.as_mut().unwrap();
			let mut rng = rand::thread_rng();
			let op:i32 = if self.pid == 1 { 2 } else { 1 }; //other player
			
			//set reward (if draw, reward already set properly)
			if state == self.pid { self.memreward = REW_WIN; }
			else if state == op { self.memreward = REW_LOSE; }
			
			//end-values of network should meet reward exactly
			self.memqval[self.memplay as usize] = self.memreward;
			
			//train on experience replay and the latest experience (q update)
			let mut trainingset = Vec::new();
			//experience
			if exp_buffer.len() > 0
			{
				for _ in 0..EXP_REP_BATCH
				{ //EXP_REP_BATCH experiences to replay
					let repindex = rng.gen::<usize>() % exp_buffer.len();
					let mut qval = nn.run(&exp_buffer[repindex].0); //.0 = state 1
					if exp_buffer[repindex].2 == REW_LOSE || exp_buffer[repindex].2 == REW_WIN //.2 = reward
					{
						qval[exp_buffer[repindex].1] = exp_buffer[repindex].2; //.1 = action, .2 = reward
					}
					else
					{
						let qval2 = targetnn.run(&exp_buffer[repindex].3); //.3 = state 2
						let max = qval2[PlayerAIQ::argmax(&qval2) as usize];
						qval[exp_buffer[repindex].1] = (exp_buffer[repindex].2 + GAMMA * max) / (1.0 + GAMMA); //.1 = action, .2 = reward
					}
					trainingset.push((exp_buffer[repindex].0.clone(), qval));
				}
			}
			//latest
			trainingset.push((self.memstate.clone(), self.memqval.clone()));
			nn.train(&trainingset)
				.halt_condition(HaltCondition::Epochs(EPOCHS))
				.log_interval(None)
				//.log_interval(Some(2)) //debug
				.momentum(MOM)
				.rate(self.lr)
				.lambda(LAMBDA / (self.games_played as f64 + 1.0))
				.go();
			//save latest as experience if not draw (would cause difficulties and is not as important)
			if self.memreward != REW_NORMAL
			{
				if exp_buffer.len() >= EXP_REP_SIZE
				{
					exp_buffer.remove(0); //remove first element
				}
				let (state1, _) = trainingset.pop().unwrap();
				exp_buffer.push((state1, self.memplay as usize, self.memreward, Vec::new())); //state1 = memstate (so state to choose action), new vec for empty end state
			}
		}
		
		//parameters
		self.games_played += 1;
		self.lr = self.get_lr();
		self.exploration = self.get_exploration();
		self.targetnn = self.nn.clone();
		self.memreward = REW_FLAG;
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
			let res2 = writeln!(&mut writer, "{}", self.nn.as_mut().unwrap().to_json());
			let res3 = write!(&mut writer, "{}", json::encode(self.exp_buffer.as_mut().unwrap()).unwrap());
			if res1.is_err() || res2.is_err() || res3.is_err() { println!("Warning: There was an error while writing AIQ NN file!"); return; }
		}
	}
}
