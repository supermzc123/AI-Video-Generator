from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

SERVICE_NAME = "io.github.supermzc123.aivideogenerator"
ACCOUNT_NAME = "llm-api-key"


class CredentialStoreError(RuntimeError):
    pass


if os.name == "nt":
    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2
    _ERROR_NOT_FOUND = 1168

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    _advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    _cred_write = _advapi32.CredWriteW
    _cred_write.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
    _cred_write.restype = wintypes.BOOL
    _cred_read = _advapi32.CredReadW
    _cred_read.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
    ]
    _cred_read.restype = wintypes.BOOL
    _cred_delete = _advapi32.CredDeleteW
    _cred_delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    _cred_delete.restype = wintypes.BOOL
    _cred_free = _advapi32.CredFree
    _cred_free.argtypes = [ctypes.c_void_p]


def _target(account: str = ACCOUNT_NAME) -> str:
    return f"{SERVICE_NAME}/{account}"


def store_secret(value: str, *, account: str = ACCOUNT_NAME) -> None:
    if os.name != "nt":
        raise CredentialStoreError(
            "系统凭据库仅在Windows桌面端可写；Linux请使用环境变量或systemd credential"
        )
    if not value:
        raise CredentialStoreError("credential value must not be empty")
    encoded = value.encode("utf-16-le")
    blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
    credential = _CREDENTIALW(
        Type=_CRED_TYPE_GENERIC,
        TargetName=_target(account),
        CredentialBlobSize=len(encoded),
        CredentialBlob=ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte)),
        Persist=_CRED_PERSIST_LOCAL_MACHINE,
        UserName=account,
    )
    if not _cred_write(ctypes.byref(credential), 0):
        raise CredentialStoreError(
            f"Windows Credential Manager写入失败：{ctypes.get_last_error()}"
        )


def load_secret(*, account: str = ACCOUNT_NAME) -> str | None:
    if os.name != "nt":
        return None
    pointer = ctypes.POINTER(_CREDENTIALW)()
    if not _cred_read(_target(account), _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        error = ctypes.get_last_error()
        if error == _ERROR_NOT_FOUND:
            return None
        raise CredentialStoreError(f"Windows Credential Manager读取失败：{error}")
    try:
        credential = pointer.contents
        if not credential.CredentialBlob or not credential.CredentialBlobSize:
            return None
        raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return raw.decode("utf-16-le")
    finally:
        _cred_free(pointer)


def delete_secret(*, account: str = ACCOUNT_NAME) -> None:
    if os.name != "nt":
        return
    if not _cred_delete(_target(account), _CRED_TYPE_GENERIC, 0):
        error = ctypes.get_last_error()
        if error != _ERROR_NOT_FOUND:
            raise CredentialStoreError(f"Windows Credential Manager删除失败：{error}")
