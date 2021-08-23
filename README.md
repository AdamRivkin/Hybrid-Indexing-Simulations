# Hybrid-Indexing-Simulations

This is the code upload for the simulations from the paper Improved Structured Encryption for SQL Databases via Hybrid Indexing. The full paper can be found at https://eprint.iacr.org/2021/852.pdf and a preliminary version of the paper was published at ACNS 2021.

## Running the simulations

You can run a test harness of the simulations through the following command: python3 compare_joins.py

The City of Chicago's tables should be placed in the folder with the path "chicago15". They originate from https://data.cityofchicago.org/

The Sakila database's tables should be placed in the folder with the path "sakila-db/sakila-csv". They originate from https://dev.mysql.com/doc/sakila/en/

The test harness supports the following commands:
help         - prints the commands
loadTables   - loads in the Sakila and Chicago datasets
tableInfo    - gets info on the loaded datasets"
singleJoins  - runs experiments on single attribute joins
getNonEmpty  - creates pickle files with the non-empty single attribute joins. Needed to generate queries
genQueries   - generates 200 random queries from the Sakila and Chicago datasets
runQueries   - generates a summary from the 200 random queries
printSummary - prints a summary of the results from running 200 queries
q            - quits the test harness

The code was written and last tested in python 3.8.6.
