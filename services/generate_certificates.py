from __future__ import annotations

import argparse
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from core.comms.config import PROJECT_ROOT, load_protocol_config, to_mapping


def certificate_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """解析 OPC UA 证书路径。"""

    certificate_config = config["opcua"]["certificates"]
    directory = PROJECT_ROOT / str(certificate_config["directory"])
    return {
        "directory": directory,
        "server_certificate": directory
        / str(certificate_config["server_certificate"]),
        "server_private_key": directory
        / str(certificate_config["server_private_key"]),
        "client_certificate": directory
        / str(certificate_config["client_certificate"]),
        "client_private_key": directory
        / str(certificate_config["client_private_key"]),
    }


def ensure_certificates(
    config: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Path]:
    """生成包含 URI、localhost 与 127.0.0.1 SAN 的应用证书。"""

    paths = certificate_paths(config)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    server_uri = str(config["opcua"]["application_uri"])
    client_uri = "urn:apal:cps:opcua:client"
    _generate_application_certificate(
        certificate_path=paths["server_certificate"],
        key_path=paths["server_private_key"],
        common_name="APAL CPS OPC UA Server",
        application_uri=server_uri,
        extended_usage=ExtendedKeyUsageOID.SERVER_AUTH,
        force=force,
    )
    _generate_application_certificate(
        certificate_path=paths["client_certificate"],
        key_path=paths["client_private_key"],
        common_name="APAL CPS OPC UA Client",
        application_uri=client_uri,
        extended_usage=ExtendedKeyUsageOID.CLIENT_AUTH,
        force=force,
    )
    return paths


def _generate_application_certificate(
    certificate_path: Path,
    key_path: Path,
    common_name: str,
    application_uri: str,
    extended_usage: x509.ObjectIdentifier,
    force: bool,
) -> None:
    if not force and certificate_path.is_file() and key_path.is_file():
        return

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "APAL CPS Course Project"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        ]
    )
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.UniformResourceIdentifier(application_uri),
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=True,
                data_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([extended_usage]), critical=False)
        .sign(private_key=private_key, algorithm=hashes.SHA256())
    )

    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.DER))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 OPC UA 应用证书")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    parser.add_argument("--force", action="store_true", help="覆盖现有证书")
    args = parser.parse_args()
    config = to_mapping(load_protocol_config(args.config))
    paths = ensure_certificates(config, force=args.force)
    print(f"证书目录: {paths['directory']}")


if __name__ == "__main__":
    main()

