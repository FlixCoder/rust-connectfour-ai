mod game;

use game::*;
use std::time::Instant;

const G_P_S:u32 = 100; //games per side


fn main()
{
	let module = 7;
	let num = 1_000_000;
	
	match module
	{
		0 => test_io(true), //learn
		1 => test_io(false), //dont learn
		2 => test_random(num),
		3 => test_minimax(num),
		4 => learn_random(num),
		5 => learn_minimax(num),
		6 => learn_ai(num),
		7 => for _ in 0..10 { learn_ai(num/10); },
		_ => (),
	}
}

fn test_io(learn:bool)
{
	println!("Player 1: IO");
	println!("Player 2: AIQ{}", if learn {""} else {"Fixed"});
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::IO);
	if learn { game.set_player2(PlayerType::AIQ); }
	else { game.set_player2(PlayerType::AIQFixed); }
	
	game.play_many(2, 1);
}

fn test_random(num:u32)
{
	println!("Player 1: Random");
	println!("Player 2: AIQFixed");
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::Random);
	game.set_player2(PlayerType::AIQFixed);
	
	let now = Instant::now();
	game.play_many(num, G_P_S);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

fn test_minimax(num:u32)
{
	println!("Player 1: Minimax");
	println!("Player 2: AIQFixed");
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::Minimax);
	game.set_player2(PlayerType::AIQFixed);
	
	let now = Instant::now();
	game.play_many(num, G_P_S);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

fn learn_random(num:u32)
{
	println!("Player 1: Random");
	println!("Player 2: AIQ");
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::Random);
	game.set_player2(PlayerType::AIQ);
	
	let now = Instant::now();
	game.play_many(num, G_P_S);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

fn learn_minimax(num:u32)
{
	println!("Player 1: Minimax");
	println!("Player 2: AIQ");
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::Minimax);
	game.set_player2(PlayerType::AIQ);
	
	let now = Instant::now();
	game.play_many(num, G_P_S);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}

fn learn_ai(num:u32)
{
	println!("Player 1: AIQFixed");
	println!("Player 2: AIQ");
	
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::AIQFixed);
	game.set_player2(PlayerType::AIQ);
	
	let now = Instant::now();
	game.play_many(num, G_P_S);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!("");
}
