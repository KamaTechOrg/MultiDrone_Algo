# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: stereo_vehicle.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'stereo_vehicle.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14stereo_vehicle.proto\x12\x0estereo_vehicle\"\x8d\x01\n\rHeightRequest\x12\x12\n\nleft_image\x18\x01 \x01(\x0c\x12\x13\n\x0bright_image\x18\x02 \x01(\x0c\x12\t\n\x01x\x18\x03 \x01(\x05\x12\t\n\x01y\x18\x04 \x01(\x05\x12\x14\n\x0c\x66ocal_length\x18\x05 \x01(\x02\x12\x10\n\x08\x62\x61seline\x18\x06 \x01(\x02\x12\x15\n\rcamera_height\x18\x07 \x01(\x02\"6\n\x16IdentifyVehicleRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\r\n\x05video\x18\x02 \x01(\x0c\"1\n\x0e\x43ontrolRequest\x12\x10\n\x08\x61\x63tivate\x18\x01 \x01(\x08\x12\r\n\x05video\x18\x02 \x01(\x0c\"\"\n\x0f\x43ontrolResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\" \n\x0eNumberResponse\x12\x0e\n\x06number\x18\x01 \x01(\x02\"\'\n\x14\x42ooleanArrayResponse\x12\x0f\n\x07results\x18\x01 \x03(\x08\"\x07\n\x05\x45mpty\"\x1e\n\rFrameResponse\x12\r\n\x05\x66rame\x18\x01 \x01(\x0c\x32\xdd\x02\n\x0fImageProcessing\x12K\n\nFindHeight\x12\x1d.stereo_vehicle.HeightRequest\x1a\x1e.stereo_vehicle.NumberResponse\x12_\n\x0fIdentifyVehicle\x12&.stereo_vehicle.IdentifyVehicleRequest\x1a$.stereo_vehicle.BooleanArrayResponse\x12P\n\rControlThread\x12\x1e.stereo_vehicle.ControlRequest\x1a\x1f.stereo_vehicle.ControlResponse\x12J\n\x12GetStabilizedFrame\x12\x15.stereo_vehicle.Empty\x1a\x1d.stereo_vehicle.FrameResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'stereo_vehicle_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_HEIGHTREQUEST']._serialized_start=41
  _globals['_HEIGHTREQUEST']._serialized_end=182
  _globals['_IDENTIFYVEHICLEREQUEST']._serialized_start=184
  _globals['_IDENTIFYVEHICLEREQUEST']._serialized_end=238
  _globals['_CONTROLREQUEST']._serialized_start=240
  _globals['_CONTROLREQUEST']._serialized_end=289
  _globals['_CONTROLRESPONSE']._serialized_start=291
  _globals['_CONTROLRESPONSE']._serialized_end=325
  _globals['_NUMBERRESPONSE']._serialized_start=327
  _globals['_NUMBERRESPONSE']._serialized_end=359
  _globals['_BOOLEANARRAYRESPONSE']._serialized_start=361
  _globals['_BOOLEANARRAYRESPONSE']._serialized_end=400
  _globals['_EMPTY']._serialized_start=402
  _globals['_EMPTY']._serialized_end=409
  _globals['_FRAMERESPONSE']._serialized_start=411
  _globals['_FRAMERESPONSE']._serialized_end=441
  _globals['_IMAGEPROCESSING']._serialized_start=444
  _globals['_IMAGEPROCESSING']._serialized_end=793
# @@protoc_insertion_point(module_scope)
