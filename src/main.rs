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
				for i in 0..100
				{
					println!("Training {}:", i+1);
					general_play(PlayerType::AIValueFixed, PlayerType::AIValue, 100, 10, true); //train, learn
					println!("Test {}:", i+1);
					general_play(PlayerType::Minimax, PlayerType::AIValueFixed, 2, 1, true); //test with minimax
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
	let mut p1 = PlayerType::IO;
	let mut p2 = PlayerType::Minimax;
	let mut num = 2;
	let mut player1starts = true;
	
	for (i, arg) in args.enumerate()
	{
		let param = arg.trim().to_lowercase();
		match i
		{
			1 => {
					let player = string_to_player(&param);
					if player.is_some() { p1 = player.unwrap(); }
				},
			2 => {
					let player = string_to_player(&param);
					if player.is_some() { p2 = player.unwrap(); }
				},
			3 => {
					let parsed = param.parse::<u32>();
					if parsed.is_ok() { num = parsed.unwrap(); }
				},
			4 => {
					if param == "false" { player1starts = false; }
				},
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

fn string_to_player(str:&str) -> Option<PlayerType>
{
	match str
	{
		"io" => Some(PlayerType::IO),
		"random" => Some(PlayerType::Random),
		"minimax" => Some(PlayerType::Minimax),
		"aiq" => Some(PlayerType::AIQ),
		"aiqfixed" => Some(PlayerType::AIQFixed),
		"aiqplay" => Some(PlayerType::AIQPlay),
		"aivalue" => Some(PlayerType::AIValue),
		"aivaluefixed" => Some(PlayerType::AIValueFixed),
		//"aiqoff" => Some(PlayerType::AIQOff),
		//"aiqofffixed" => Some(PlayerType::AIQOffFixed),
		_ => None,
	}
}
