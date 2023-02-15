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
INTO 'results.csv' ON CLIENT USING DELIMITERS ',', '\n', '"';