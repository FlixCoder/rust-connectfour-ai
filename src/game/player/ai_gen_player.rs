#![allow(dead_code)]

use super::Player;
use super::super::field::Field;


pub struct PlayerAIGen
{
	initialized: bool,
	pid: i32, //player ID
}

impl PlayerAIGen
{
	pub fn new() -> Box<PlayerAIGen>
	{
		Box::new(PlayerAIGen { initialized: false, pid: 0 })
	}
}

impl Player for PlayerAIGen
{
	#[allow(unused_variables)]
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		self.initialized = true;
		self.pid = p;
		
		false
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		
		false
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//nothing
	}
}

impl Drop for PlayerAIGen
{
	fn drop(&mut self)
	{
		//nothing to do
	}
}
