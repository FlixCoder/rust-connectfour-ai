#![allow(dead_code)]

extern crate rand;

use self::rand::Rng;
use super::Player;
use super::super::field::Field;


pub struct PlayerRandom
{
	initialized: bool,
	pid: i32, //player ID
}

impl PlayerRandom
{
	pub fn new() -> Box<PlayerRandom>
	{
		Box::new(PlayerRandom { initialized: false, pid: 0 })
	}
}

impl Player for PlayerRandom
{
	#[allow(unused_variables)]
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		self.initialized = true;
		self.pid = p;
		true
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		
		let w = field.get_w();
		let mut rng = rand::thread_rng();
		let mut random = rng.gen::<u32>() % w;
		while !field.play(self.pid, random)
		{
			random = rng.gen::<u32>() % w;
		}
		
		true
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//nothing
	}
}

impl Drop for PlayerRandom
{
	fn drop(&mut self)
	{
		//nothing to do
	}
}
