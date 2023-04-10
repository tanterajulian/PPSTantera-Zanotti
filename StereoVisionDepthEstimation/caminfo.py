#!/usr/bin/env python3

import ctypes
import fcntl

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_WRITE = 1
_IOC_READ = 2

V4L2_CTRL_CLASS_CAMERA = 0x009a0000
V4L2_CID_CAMERA_CLASS_BASE = V4L2_CTRL_CLASS_CAMERA | 0x900
V4L2_CID_EXPOSURE_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 1
V4L2_CID_FOCUS_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 12

def _IOC(dir_, type_, nr, size):
    return (
        ctypes.c_int32(dir_ << _IOC_DIRSHIFT).value |
        ctypes.c_int32(ord(type_) << _IOC_TYPESHIFT).value |
        ctypes.c_int32(nr << _IOC_NRSHIFT).value |
        ctypes.c_int32(size << _IOC_SIZESHIFT).value)


def _IOC_TYPECHECK(t):
    return ctypes.sizeof(t)

def _IOWR(type_, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, _IOC_TYPECHECK(size))


class v4l2_control(ctypes.Structure):
    _fields_ = [
        ('id', ctypes.c_uint32),
        ('value', ctypes.c_uint32),
    ]

VIDIOC_G_CTRL = _IOWR('V', 27, v4l2_control)

exposure = v4l2_control(V4L2_CID_EXPOSURE_AUTO)
focus_auto = v4l2_control(V4L2_CID_FOCUS_AUTO)
with open('/dev/video2', 'r') as vd:
    fcntl.ioctl(vd, VIDIOC_G_CTRL, exposure)
    fcntl.ioctl(vd, VIDIOC_G_CTRL, focus_auto)
print("exposure auto: %i" % exposure.value)
print("focus auto   : %i" % focus_auto.value)
