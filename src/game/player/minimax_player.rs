#![allow(dead_code)]


use super::Player;
use super::super::field::Field;


pub struct PlayerMinimax
{
	initialized: bool,
	pid: i32, //player ID
}

impl PlayerMinimax
{
	pub fn new() -> Box<PlayerMinimax>
	{
		Box::new(PlayerMinimax { initialized: false, pid: 0 })
	}
}

impl Player for PlayerMinimax
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
		
		//do alpha beta minimax
		
		false
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//nothing
	}
}

impl Drop for PlayerMinimax
{
	fn drop(&mut self)
	{
		//nothing to do
	}
}
