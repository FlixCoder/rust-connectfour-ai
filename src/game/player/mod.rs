#![allow(dead_code)]

pub mod io_player;
pub mod random_player;
pub mod ai_q_player;
pub mod ai_gen_player;

use super::field::Field;


pub trait Player:Drop
{
	fn init(&mut self, field:&Field, p_id:i32) -> bool;
	fn play(&mut self, field:&mut Field) -> bool;
	fn outcome(&mut self, field:&mut Field, state:i32);
}
