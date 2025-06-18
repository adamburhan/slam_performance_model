from preprocessing import discretize_and_save_to_csv

# The directory where rpe errors are stored for all the 11 kitti sequences
errors_dir = '/media/adam/T9/slam_performance_model/data/errors'

train_files = [f'{errors_dir}/{i:02d}.txt' for i in range(9)] # sequences 00 to 08
test_files = [f'{errors_dir}/{i:02d}.txt' for i in range(9, 11)] # sequences 09 and 10

output_train_files = [f'{errors_dir}/discretized/{i:02d}.csv' for i in range(9)]
output_test_files = [f'{errors_dir}/discretized/{i:02d}.csv' for i in range(9, 11)]

discretize_and_save_to_csv(train_files, test_files, output_train_files, output_test_files, n_quantiles=5)