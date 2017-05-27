#![allow(dead_code)]

mod field;
mod player;

use self::field::Field;
use self::player::Player;
use self::player::io_player::PlayerIO;
use self::player::random_player::PlayerRandom;
use self::player::ai_q_player::PlayerAIQ;
use self::player::ai_gen_player::PlayerAIGen;


pub enum PlayerType {None, IO, Random, AIQ, AIQFixed, AIGen}

pub struct Game
{
	field: Field,
	p1: Option<Box<Player>>,
	p2: Option<Box<Player>>,
	startp: u32
}

impl Game
{
	pub fn new() -> Game
	{
		Game { field: Field::new(7, 6), p1: None, p2: None, startp: 1 }
	}
	
	pub fn set_player1(&mut self, p:PlayerType) -> bool
	{
		match p
		{
			PlayerType::None => self.p1 = None,
			PlayerType::IO => self.p1 = Some(PlayerIO::new()),
			PlayerType::Random => self.p1 = Some(PlayerRandom::new()),
			PlayerType::AIQ => self.p1 = Some(PlayerAIQ::new(false)),
			PlayerType::AIQFixed => self.p1 = Some(PlayerAIQ::new(true)),
			PlayerType::AIGen => self.p1 = Some(PlayerAIGen::new()),
		}
		
		if self.p1.is_some()
		{
			if !self.p1.as_mut().unwrap().init(&self.field, 1)
			{
				self.p1 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_player2(&mut self, p:PlayerType) -> bool
	{
		match p
		{
			PlayerType::None => self.p2 = None,
			PlayerType::IO => self.p2 = Some(PlayerIO::new()),
			PlayerType::Random => self.p2 = Some(PlayerRandom::new()),
			PlayerType::AIQ => self.p2 = Some(PlayerAIQ::new(false)),
			PlayerType::AIQFixed => self.p2 = Some(PlayerAIQ::new(true)),
			PlayerType::AIGen => self.p2 = Some(PlayerAIGen::new()),
		}
		
		if self.p2.is_some()
		{
			if !self.p2.as_mut().unwrap().init(&self.field, 2)
			{
				self.p2 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_start_player(&mut self, p:u32) -> bool
	{
		if p < 1 || p > 2 { return false; }
		self.startp = p;
		true
	}
	
	pub fn is_ready(&self) -> bool
	{
		self.p1.is_some() && self.p2.is_some()
	}
	
	pub fn play(&mut self) -> bool
	{
		if !self.is_ready() { return false; }
		
		let p1 = self.p1.as_mut().unwrap();
		let p2 = self.p2.as_mut().unwrap();
		
		self.field.reset();
		let mut turn1:bool = self.startp == 1;
		let mut state = 0;
		
		while state == 0
		{
			if turn1
			{
				if !p1.play(&mut self.field)
				{ println!("Warning: player 1 did not play!"); }
				turn1 = !turn1;
			}
			else
			{
				if !p2.play(&mut self.field)
				{ println!("Warning: player 2 did not play!"); }
				turn1 = !turn1;
			}
			state = self.field.get_state();
			//self.field.print(); //debug
		}
		
		p1.outcome(&mut self.field, state);
		p2.outcome(&mut self.field, state);
		
		true
	}
	
	pub fn play_many(&mut self, num:u32) -> bool
	{
		if num<1 { return false; }
		
		let mut p1win:u32 = 0;
		let mut draw:u32 = 0;
		let mut p2win:u32 = 0;
		
		for i in 0..num
		{
			self.set_start_player((i % 2) + 1); //switch sides every game
			if !self.play() { return false; }
			match self.field.get_state()
			{
				-1 => draw += 1,
				1 => p1win += 1,
				2 => p2win += 1,
				_ => println!("Warning: game ended running!"),
			}
		}
		
		let p1wr:f64 = (p1win as f64)/(num as f64)*100.0;
		let drawrate:f64 = (draw as f64)/(num as f64)*100.0;
		let p2wr:f64 = (p2win as f64)/(num as f64)*100.0;
		
		println!("---------------------------------");
		println!("Results:");
		println!("Player X wins: {:>6.2}% ({}/{})", p1wr, p1win, num);
		println!("Draws:         {:>6.2}% ({}/{})", drawrate, draw, num);
		println!("Player O wins: {:>6.2}% ({}/{})", p2wr, p2win, num);
		
		true
	}
}
