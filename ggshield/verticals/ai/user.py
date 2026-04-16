from typing import Optional

from pygitguardian.models import UserInfo


def get_user_info(machine_id: Optional[str] = None) -> UserInfo:
    """Dummy implementation. To replace in next commit."""
    return UserInfo(
        hostname="host",
        username="toto",
        machine_id=machine_id or "mid",
        user_email="toto@gg.com",
    )
