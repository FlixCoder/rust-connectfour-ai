mod game;

use game::*;
use std::time::Instant;
use std::env;


#[allow(unreachable_code)]
fn main()
{
	match 5
	{
		0 => general_play(PlayerType::IO, PlayerType::AIValue, 2, 1, true), //play against IO + learn
		1 => general_play(PlayerType::IO, PlayerType::AIValueFixed, 2, 1, true), //test with IO
		2 => general_play(PlayerType::Minimax, PlayerType::AIValueFixed, 100, 1, true), //test with minimax
		3 => general_play(PlayerType::Random, PlayerType::AIValueFixed, 1000, 1, true), //test with random
		4 => general_play(PlayerType::AIValueFixed, PlayerType::AIValue, 1_000, 10, true), //training
		5 => { //continuous training and testing
				println!("Training:");
				for i in 0..10
				{
					println!("Training {}:", i+1);
					general_play(PlayerType::AIValueFixed, PlayerType::AIValue, 1_000, 10, true); //train, learn
					println!("Test {}:", i+1);
					general_play(PlayerType::Minimax, PlayerType::AIValueFixed, 100, 1, true); //test with minimax
				}
				println!("Testing:");
				general_play(PlayerType::Random, PlayerType::AIValueFixed, 1000, 1, true); //test with random
				general_play(PlayerType::IO, PlayerType::AIValueFixed, 2, 1, true); //test with IO
			},
		_ => {
				//general playing with command line arguments
				play_from_args();
			}
	}
}


#[allow(dead_code)]
fn play_from_args()
{
	let args = env::args();
	//general playing with command line arguments
	let p1 = PlayerType::IO;
	let p2 = PlayerType::Minimax;
	let num = 2;
	let player1starts = true;
	
	for (i, arg) in args.enumerate()
	{
		match i
		{
			1 => {},
			2 => {},
			3 => {},
			4 => {},
			_ => {}, //ignore first and all other args
		}
	}
	
	println!("Running:");
	general_play(p1, p2, num, 1, player1starts);
}

#[allow(dead_code)]
fn general_play(p1:PlayerType, p2:PlayerType, num:u32, gps:u32, player1starts:bool)
{
	println!("Player X: {:?}", p1);
	println!("Player O: {:?}", p2);
	println!("Playing {} games..", num);
	
	//prepare
	let mut game = Game::new();
	game.set_start_player(if player1starts {1} else {2});
	game.set_player1(p1);
	game.set_player2(p2);
	
	//measure time
	let now = Instant::now();
	let (p1w, p2w) = game.play_many(num, gps); //play
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
	
	//drop worse player first in case 2 learning agents play against each other and use the same file
	if p1w > p2w
	{
		game.set_player2(PlayerType::None);
		game.set_player1(PlayerType::None);
	}
	else
	{
		game.set_player1(PlayerType::None);
		game.set_player2(PlayerType::None);
	}
}

