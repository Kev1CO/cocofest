# Florine's contribution to Cocofest identification examples

## Force identification
2 Python files :
- `from_exp_model_id.py`: the basic file to run the identification
- `from_exp_model_id_LOO.py`: the file to automatically run the identification optimization using Leave One Out method.

How to use `from_exp_model_id_LOO.py` ? \
You can either run the optimization with all frequencies `optim_all_concat()` or per frequency `optim_per_freq()`.\
To run automatically the optimization, you can use the `id_auto()` function. You need to specify the participant number you want, the method (all freq or per freq) and the muscles you want to use to identify the parameters.\
After automatically running optimization, you can easily check the results by using the `check_data_id()` function. You only have to specify the participant numbers you want to check, the muscle names and the method (all freq or per freq).\
Then, you can use the `compute_out()` function, to compute the RMSE between the force generated from the identified parameters and the experimental force (for the test train).
You can run it automatically by using `loo_auto()`, then plot the results to check them using `check_data_loo()`.

## Motion identification