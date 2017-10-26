//! trained NN to represent value heuristic and use minimax
#![allow(dead_code)]

extern crate rand;
extern crate nn;
extern crate rustc_serialize;

use self::rustc_serialize::json;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use self::rand::Rng;
use self::nn::{NN, HaltCondition, Activation};
use super::Player;
use super::super::field::Field;
use std::f64;

const DEEPNESS:u32 = 5; //recursion limit
const LEARN_FREQ:u32 = 10; //number of games between learning to collect data to train with
const KEEP_NUM:usize = 0; //number of state value pairs to save between learning episodes
const GAMMA:f64 = 0.9; //temporal unsureness factor
const LR:f64 = 0.01; //neural net learning rate (deterministic -> high)
const LR_DECAY:f64 = 0.001 / 1000f64; //NN learning rate decrease per game(s)
const LR_MIN:f64 = 0.001; //minimum NN LR
const LAMBDA:f64 = 0.001; //L2 regularization parameter lambda (divide by n manually, pick very small > 0, like pick LAMBDA / n)
const MOM:f64 = 0.5; //neural net momentum
const EPOCHS:u32 = 2; //NN training epochs per data set

//values for a won or lost game in minimax and heuristic (neural net outputs should be a lot closer to zero)
const VAL_MAX:f64 = 10002.0; //f64::MAX
const VAL_MIN:f64 = -10002.0; //f64::MIN
//values for value to train with NN
const VAL_WIN:f64 = 0.9; //starting player wins value for NN learning (greater than 0!)
const VAL_DRAW:f64 = 0.0; //draw's value for NN learning (0!)
const VAL_LOSE:f64 = -0.9; //starting player loses value for NN learning (less than 0!)

pub struct PlayerAIValue
{
	initialized: bool,
	fixed: bool, //fixed agent? (don't learn)
	pid: i32, //player ID
	startp: i32, //starting player
	games_played: u32, //number of games the agent played
	filename: String, //file name for NN/agent information
	nn: Option<NN>, //neural network for neutral state evaluation (value based on starting player)
	lr: f64, //NN learning rate
	current_game: Vec<Vec<f64>>, //buffer for states, that occured in the current game
	games_buffer: Vec<(Vec<f64>,Vec<f64>)>, //buffer of game data to learn -> training buffer
}

impl PlayerAIValue
{
	pub fn new(fix: bool) -> Box<PlayerAIValue>
	{
		Box::new(PlayerAIValue { initialized: false, fixed: fix, pid: 0, startp: 0, games_played: 0,
						filename: String::new(), nn: None, lr: LR, current_game: Vec::new(),
						games_buffer: Vec::new() })
	}
	
	fn get_lr(&self) -> f64
	{
		LR_MIN.max(LR - LR_DECAY * self.games_played as f64)
	}
	
	//raw field
	fn field_to_input(field:&mut Field, p:i32) -> Vec<f64>
	{ //input: p = start player
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity(field.get_size() as usize);
		//1 nodes for every square: -1 enemy, 0 free, 1 own
		for val in field.get_field().iter()
		{
			if *val == p { input.push(1f64); }
			else if *val == op { input.push(-1f64); }
			else { input.push(0f64); } //empty square
		}
		//return
		input
	}
	
	//returns value of board position: +1.0 player wins, -1.0 other player wins, 0.0 draw or even board
	fn heur(&self, field:&mut Field, p:i32, deep:u32) -> f64 //p = player. translated from start player by (value * -1) if they are not same.
	{
		let op = if p == 1 {2} else {1};
		let state = field.get_state(); //return best or worst value on win/loose (neutral on tie)
		if state == -1 { return 0.0; }
		else if state == p { return VAL_MAX - deep as f64; }
		else if state == op { return VAL_MIN + deep as f64; }
		else
		{ //game running -> evaluate
			let nn = self.nn.as_ref().unwrap();
			let state = PlayerAIValue::field_to_input(field, self.startp);
			let factor = if p == self.startp { 1.0 } else { -1.0 }; //if p is not startp value has to be reversed
			
			let result = nn.run(&state);
			let value = factor * result[0];
			
			return value;
		}
	}
	
	fn minimax(&self, field:&mut Field, p:i32, deep:u32) -> f64
	{
		let op = if p == 1 {2} else {1};
		if deep > DEEPNESS { return self.heur(field, if deep%2 == 0 {op} else {p}, deep); } //leaf node -> return evaluated heuristic, mechanism to get heur always for same player
		let state = field.get_state(); //return early on game end
		if state == -1 { return 0.0; }
		else if state == p { return if deep%2 == 0 {VAL_MIN + deep as f64} else {VAL_MAX - deep as f64}; }
		else if state == op { return if deep%2 == 0 {VAL_MAX - deep as f64} else {VAL_MIN + deep as f64}; }
		
		//else: game running -> go deeper
		let mut heur = if deep%2 == 0 { f64::INFINITY } else { f64::NEG_INFINITY };
		for i in 0..field.get_w()
		{
			if field.play(p, i)
			{
				let val = self.minimax(field, op, deep+1);
				field.undo();
				if (deep%2 == 0 && val < heur) || (deep%2 == 1 && val > heur) //min or max according to which player's turn it is
				{
					heur = val;
				}
			}
		}
		heur
	}
	
	fn learn_from_data(&mut self)
	{
		//use the collected data to improve the neural net
		let nn = self.nn.as_mut().unwrap();
		let mut rng = rand::thread_rng();
		
		//shuffle data
		let len = self.games_buffer.len();
		for _ in 0..len
		{ //n random O(1) operations on the buffer to shuffle
			let i = rng.gen::<usize>() % len;
			let item = self.games_buffer.swap_remove(i);
			self.games_buffer.push(item);
		}
		
		//learn
		nn.train(&self.games_buffer)
			.halt_condition(HaltCondition::Epochs(EPOCHS))
			.log_interval(None)
			.momentum(MOM)
			.rate(self.lr)
			.lambda(LAMBDA / (self.games_played as f64 + 1000.0))
			.go();
		
		//flush buffer
		self.games_buffer.truncate(KEEP_NUM); //.clear();
	}
}

impl Player for PlayerAIValue
{
	#[allow(unused_variables)]
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		if DEEPNESS < 1 { return false; } //invalid player, could cause bugs else
		
		self.pid = p;
		
		self.filename = format!("AIValue-{}x{}.NN", field.get_w(), field.get_h());
		let file = File::open(&self.filename);
		if file.is_err()
		{
			//create new neural net, as it could not be loaded
			let n = field.get_size();
			self.nn = Some(NN::new(&[n, 6*n, 3*n, n, 1], Activation::PELU, Activation::Tanh)); //set size of NN layers here, be careful with activation function
			//games_played, lr already set
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
			self.games_buffer = json::decode(&exps).unwrap();
			
			self.lr = self.get_lr();
		}
		
		self.initialized = true;
		true
	}
	
	#[allow(unused_variables)]
	fn startp(&mut self, p:i32)
	{
		self.startp = p;
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		
		let p = self.pid;
		let op = if p == 1 {2} else {1};
		
		//decide which action x to take
		let mut x:u32 = 0;
		let mut max = f64::NEG_INFINITY;
		//decide by evaluation
		for i in 0..field.get_w()
		{
			let mut val = f64::NEG_INFINITY;
			if field.play(p, i)
			{
				val = self.minimax(field, op, 2);
				field.undo();
			}
			if max < val || !field.is_valid_play(x)
			{
				max = val;
				x = i;
			}
		}
		
		//save game state for later mapping to win/loose. only if not fixed to save memory
		if !self.fixed
		{
			let state = PlayerAIValue::field_to_input(field, self.startp);
			self.current_game.push(state);
		}
		
		//debug
		//println!("Heur: {}", max);
		
		//play (actually should always be true, unless game was finished before method invocation)
		field.play(p, x)
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//parameters
		self.games_played += 1;
		self.lr = self.get_lr();
		
		//collect data and learn if not fixed (to save memory and computation else)
		if !self.fixed
		{
			//collect data
			let op:i32 = if self.startp == 1 { 2 } else { 1 }; //other player
			let mut temporal_factor = 1.0;
			let mut value = VAL_DRAW; //draw
			if state == self.startp { value = VAL_WIN; } //win
			else if state == op { value = VAL_LOSE; } //lose
			
			while !self.current_game.is_empty()
			{
				let state = self.current_game.pop().unwrap();
				let result = vec![temporal_factor * value];
				self.games_buffer.push((state, result));
				temporal_factor *= GAMMA; //induce unsureness at the start of the game, avoid constant value learned, prioritize fast win
			}
			
			//learn if it is time
			if self.games_played % LEARN_FREQ == 0
			{
				self.learn_from_data();
			}
		}
	}
}

impl Drop for PlayerAIValue
{
	fn drop(&mut self)
	{
		//if it was allowed to learn and was initialized:
		if self.initialized && !self.fixed
		{
			//learn from remaining data
			self.learn_from_data();
			
			//write neural net to file
			let file = File::create(&self.filename);
			if file.is_err() { println!("Warning: Could not write AIValue NN file!"); return; }
			let mut writer = BufWriter::new(file.unwrap());
			
			let res1 = writeln!(&mut writer, "{}", self.games_played);
			let res2 = writeln!(&mut writer, "{}", self.nn.as_mut().unwrap().to_json());
			let res3 = write!(&mut writer, "{}", json::encode(&self.games_buffer).unwrap());
			if res1.is_err() || res2.is_err() || res3.is_err() { println!("Warning: There was an error while writing AIValue NN file!"); return; }
		}
	}
}
