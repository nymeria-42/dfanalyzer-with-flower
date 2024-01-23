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
		itrainingconfig tc)
INTO 'training_config.csv' ON CLIENT USING DELIMITERS ',', '\n', '"';

COPY (
    SELECT
        st.server_round,
        st.total_num_clients,
        st.total_num_examples,
        st.accuracy,
        st.loss,
        st.training_time,
        st.val_accuracy,
        st.val_loss, 
        st.starting_time, 
		st.ending_time,
        se.total_num_clients,
        se.total_num_examples,
        se.accuracy,
        se.loss,
        se.evaluation_time
    FROM
        oservertrainingaggregation as st
    JOIN 
        oserverevaluationaggregation as se
        ON st.server_round = se.server_round )
INTO 'metrics_results.csv' ON CLIENT USING DELIMITERS ',', '\n', '"';

COPY (
    SELECT
        ilw.server_id, ilw.starting_time, ilw.loading_time, olw.loaded_from_mongo, sc.address, sc.num_rounds
    FROM
        iloadglobalweights as ilw
    JOIN 
        oloadglobalweights as olw
        ON ilw.id = olw.id 
    JOIN iserverconfig as sc
        on sc.server_id = ilw.server_id)
INTO 'load_weights.csv' ON CLIENT USING DELIMITERS ',', '\n', '"';

COPY (
    SELECT
        server_id, server_round, weights_mongo_id, insertion_time, checkpoint_time
    FROM
        oservertrainingaggregation)
INTO 'insert_weights.csv' ON CLIENT USING DELIMITERS ',', '\n', '"';
