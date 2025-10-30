import json
import threading
from typing import Optional, Dict, Any

from logger import logger

try:
    from kafka import KafkaProducer
except Exception:
    KafkaProducer = None  # type: ignore


class KafkaBus:
    """
    Thin Kafka wrapper with lazy init and JSON publishing.
    Safe no-op when disabled or kafka-python isn't installed.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg or {}
        self._enabled = bool(self._cfg.get("kafka", {}).get("enabled", False))
        self._lock = threading.Lock()
        self._producer: Optional[KafkaProducer] = None  # type: ignore

    def _build_producer(self) -> Optional[KafkaProducer]:  # type: ignore
        if not self._enabled:
            return None
        if KafkaProducer is None:
            logger.debug("[KAFKA] kafka-python not installed; Kafka disabled")
            return None

        kcfg = self._cfg.get("kafka", {})
        kwargs: Dict[str, Any] = {
            "bootstrap_servers": kcfg.get("bootstrap_servers", "localhost:9092"),
            "client_id": kcfg.get("client_id", "iwr6843-app"),
            "acks": int(kcfg.get("acks", 1)),
            "value_serializer": lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            "key_serializer": lambda v: None if v is None else str(v).encode("utf-8"),
        }
        comp = kcfg.get("compression_type")
        if comp:
            kwargs["compression_type"] = comp

        # Optional SSL
        sslcfg = kcfg.get("ssl", {}) or {}
        if bool(sslcfg.get("enabled")):
            kwargs.update({
                "security_protocol": "SSL",
                "ssl_cafile": sslcfg.get("cafile") or None,
                "ssl_certfile": sslcfg.get("certfile") or None,
                "ssl_keyfile": sslcfg.get("keyfile") or None,
                "ssl_check_hostname": bool(sslcfg.get("check_hostname", True)),
            })
        try:
            prod = KafkaProducer(**kwargs)
            logger.info("[KAFKA] Producer ready (%s)", kwargs.get("bootstrap_servers"))
            return prod
        except Exception as e:
            logger.warning(f"[KAFKA] init failed: {e}")
            return None

    def _get(self) -> Optional[KafkaProducer]:  # type: ignore
        if not self._enabled:
            return None
        if self._producer is not None:
            return self._producer
        with self._lock:
            if self._producer is None:
                self._producer = self._build_producer()
        return self._producer

    def publish(self, topic: str, value: dict, key: Optional[str] = None) -> bool:
        try:
            prod = self._get()
            if prod is None:
                return False
            prod.send(topic, value=value, key=key)
            return True
        except Exception as e:
            logger.debug(f"[KAFKA] publish error to {topic}: {e}")
            return False

    def flush(self, timeout: Optional[float] = None) -> None:
        try:
            prod = self._producer
            if prod is not None:
                prod.flush(timeout=timeout)
        except Exception:
            pass

    def close(self) -> None:
        try:
            prod = self._producer
            if prod is not None:
                prod.flush(timeout=1.0)
                prod.close(timeout=1.0)
        except Exception:
            pass

