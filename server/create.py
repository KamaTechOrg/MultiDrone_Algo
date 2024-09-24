from grpc_tools import protoc

protoc.main((
    '',
    '-I.',  # Input directory
    '--python_out=.',  # Output directory for generated Python code
    '--grpc_python_out=.',  # Output directory for gRPC code
    'stereo_vehicle.proto',  # Your proto file
))
