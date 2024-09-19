# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: allert_server.proto
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
    'allert_server.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13\x61llert_server.proto\x12\x06\x61lerts\"\x9c\x01\n\x11\x43ountAlertRequest\x12\r\n\x05is_on\x18\x01 \x01(\x08\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\x12\x15\n\rcoordinate1_x\x18\x03 \x01(\x05\x12\x15\n\rcoordinate1_y\x18\x04 \x01(\x05\x12\x15\n\rcoordinate2_x\x18\x05 \x01(\x05\x12\x15\n\rcoordinate2_y\x18\x06 \x01(\x05\x12\r\n\x05image\x18\x07 \x01(\x0c\"%\n\x12\x43ountAlertResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"&\n\x15SendImageAlertRequest\x12\r\n\x05is_on\x18\x01 \x01(\x08\")\n\x16SendImageAlertResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"%\n\x14\x41\x63\x63identAlertRequest\x12\r\n\x05is_on\x18\x01 \x01(\x08\"(\n\x15\x41\x63\x63identAlertResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"%\n\x14OddEventAlertRequest\x12\r\n\x05is_on\x18\x01 \x01(\x08\"(\n\x15OddEventAlertResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"$\n\x13IsEmptyAlertRequest\x12\r\n\x05is_on\x18\x01 \x01(\x08\"\'\n\x14IsEmptyAlertResponse\x12\x0f\n\x07message\x18\x01 \x01(\t2\xa3\x06\n\x0c\x41lertService\x12\x46\n\rCountAllertOn\x12\x19.alerts.CountAlertRequest\x1a\x1a.alerts.CountAlertResponse\x12G\n\x0e\x43ountAllertOff\x12\x19.alerts.CountAlertRequest\x1a\x1a.alerts.CountAlertResponse\x12Q\n\x10SendImageAlertOn\x12\x1d.alerts.SendImageAlertRequest\x1a\x1e.alerts.SendImageAlertResponse\x12R\n\x11SendImageAlertOff\x12\x1d.alerts.SendImageAlertRequest\x1a\x1e.alerts.SendImageAlertResponse\x12N\n\x0f\x41\x63\x63identAlertOn\x12\x1c.alerts.AccidentAlertRequest\x1a\x1d.alerts.AccidentAlertResponse\x12O\n\x10\x41\x63\x63identAlertOff\x12\x1c.alerts.AccidentAlertRequest\x1a\x1d.alerts.AccidentAlertResponse\x12N\n\x0fOddEventAlertOn\x12\x1c.alerts.OddEventAlertRequest\x1a\x1d.alerts.OddEventAlertResponse\x12O\n\x10OddEventAlertOff\x12\x1c.alerts.OddEventAlertRequest\x1a\x1d.alerts.OddEventAlertResponse\x12K\n\x0eIsEmptyAlertOn\x12\x1b.alerts.IsEmptyAlertRequest\x1a\x1c.alerts.IsEmptyAlertResponse\x12L\n\x0fIsEmptyAlertOff\x12\x1b.alerts.IsEmptyAlertRequest\x1a\x1c.alerts.IsEmptyAlertResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'allert_server_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_COUNTALERTREQUEST']._serialized_start=32
  _globals['_COUNTALERTREQUEST']._serialized_end=188
  _globals['_COUNTALERTRESPONSE']._serialized_start=190
  _globals['_COUNTALERTRESPONSE']._serialized_end=227
  _globals['_SENDIMAGEALERTREQUEST']._serialized_start=229
  _globals['_SENDIMAGEALERTREQUEST']._serialized_end=267
  _globals['_SENDIMAGEALERTRESPONSE']._serialized_start=269
  _globals['_SENDIMAGEALERTRESPONSE']._serialized_end=310
  _globals['_ACCIDENTALERTREQUEST']._serialized_start=312
  _globals['_ACCIDENTALERTREQUEST']._serialized_end=349
  _globals['_ACCIDENTALERTRESPONSE']._serialized_start=351
  _globals['_ACCIDENTALERTRESPONSE']._serialized_end=391
  _globals['_ODDEVENTALERTREQUEST']._serialized_start=393
  _globals['_ODDEVENTALERTREQUEST']._serialized_end=430
  _globals['_ODDEVENTALERTRESPONSE']._serialized_start=432
  _globals['_ODDEVENTALERTRESPONSE']._serialized_end=472
  _globals['_ISEMPTYALERTREQUEST']._serialized_start=474
  _globals['_ISEMPTYALERTREQUEST']._serialized_end=510
  _globals['_ISEMPTYALERTRESPONSE']._serialized_start=512
  _globals['_ISEMPTYALERTRESPONSE']._serialized_end=551
  _globals['_ALERTSERVICE']._serialized_start=554
  _globals['_ALERTSERVICE']._serialized_end=1357
# @@protoc_insertion_point(module_scope)
