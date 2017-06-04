mod game;

use game::*;
use std::time::Instant;


fn main()
{
	let num = 1_000; //number of games to play
	let gps = 100; //games per side
	let p1 = PlayerType::Minimax;
	let p2 = PlayerType::AIQOff;
	
	println!("Player X: {:?}", p1);
	println!("Player O: {:?}", p2);
	
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

