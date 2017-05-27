mod game;

use game::*;


fn main()
{
	let mut game = Game::new();
	//game.set_start_player(1);
	//game.set_player1(PlayerType::IO);
	//game.set_player1(PlayerType::Random);
	game.set_player1(PlayerType::AIQFixed);
	game.set_player2(PlayerType::AIQ);
	
	//game.play_many(2);
	game.play_many(100000);
}
