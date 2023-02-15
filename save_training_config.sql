COPY (
	SELECT
		tc.server_round,			
		tc.shuffle,
		tc.batch_size,
		tc.initial_epoch,
		tc.epochs,
		tc.steps_per_epoch, 
		tc.validation_split,
		tc.validation_batch_size
	FROM
		itrainingconfig as tc)
INTO 'training_config.csv' ON CLIENT USING DELIMITERS ',', '\n', '"';