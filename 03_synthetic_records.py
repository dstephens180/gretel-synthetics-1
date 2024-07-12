from pathlib import Path
from gretel_synthetics.config import LocalConfig

# Create a config that we can use for both training and generating data
# The default values for ``max_lines`` and ``epochs`` are optimized for training on a GPU.

config = LocalConfig(
    max_line_len=2048,   # the max line length for input training data
    vocab_size=20000,    # tokenizer model vocabulary size
    field_delimiter=",", # specify if the training text is structured, else ``None``
    overwrite=True,      # overwrite previously trained model checkpoints
    checkpoint_dir=(Path.cwd() / 'checkpoints').as_posix(),
    input_data_path="https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uber_scooter_rides_1day.csv" # filepath or S3
)



from gretel_synthetics.train import train_rnn

train_rnn(config)



# Let's generate some text!
#
# The ``generate_text`` funtion is a generator that will return
# a line of predicted text based on the ``gen_lines`` setting in your
# config.
#
# There is no limit on the line length as with proper training, your model
# should learn where newlines generally occur. However, if you want to
# specify a maximum char len for each line, you may set the ``gen_chars``
# attribute in your config object
from gretel_synthetics.generate import generate_text

# Optionally, when generating text, you can provide a callable that takes the 
# generated line as a single arg. If this function raises any errors, the 
# line will fail validation and will not be returned.  The exception message
# will be provided as a ``explain`` field in the resulting dict that gets
# created by ``generate_text``
def validate_record(line):
    rec = line.split(", ")
    if len(rec) == 6:
        float(rec[5])
        float(rec[4])
        float(rec[3])
        float(rec[2])
        int(rec[0])
    else:
        raise Exception('record not 6 parts')
        
for line in generate_text(config, line_validator=validate_record, num_lines=1000):
    print(line)







