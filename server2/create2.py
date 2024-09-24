from grpc_tools import protoc

protoc.main((
    '',
    '-I.',  # Input directory
    '--python_out=.',  # Output directory for generated Python code
    '--grpc_python_out=.',  # Output directory for gRPC code
    'proto2.proto',  # Your proto file
))
