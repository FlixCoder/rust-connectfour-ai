mod game;

use game::*;
use std::time::Instant;


#[allow(unreachable_code)]
fn main()
{
	match 3
	{
		0 => test_io(),
		1 => test(),
		2 => train(),
		3 => {
				println!("Training:");
				for i in 0..5
				{
					println!("Training {}:", i);
					train();
					println!("Testing {}:", i);
					test();
				}
				test_io();
			},
		_ => {}
	}
}


#[allow(dead_code)]
fn test_io()
{
	let num = 2; //number of games to play
	let gps = 1; //games per side
	let p1 = PlayerType::IO;
	let p2 = PlayerType::AIQFixed;
	
	println!("Player X: {:?}", p1);
	println!("Player O: {:?}", p2);
	println!("Playing {} games..", num);
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(p1);
	game.set_player2(p2);
	
	let now = Instant::now();
	game.play_many(num, gps);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

#[allow(dead_code)]
fn test()
{
	let num = 1_000; //number of games to play
	let gps = 1; //games per side
	let p1 = PlayerType::Minimax;
	let p2 = PlayerType::AIQFixed;
	
	println!("Player X: {:?}", p1);
	println!("Player O: {:?}", p2);
	println!("Playing {} games..", num);
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(p1);
	game.set_player2(p2);
	
	let now = Instant::now();
	game.play_many(num, gps);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

#[allow(dead_code)]
fn train()
{
	let num = 10_000; //number of games to play
	let gps = 10; //games per side
	let p1 = PlayerType::AIQFixed;
	let p2 = PlayerType::AIQ;
	
	println!("Player X: {:?}", p1);
	println!("Player O: {:?}", p2);
	println!("Playing {} games..", num);
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(p1);
	game.set_player2(p2);
	
	let now = Instant::now();
	game.play_many(num, gps);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

