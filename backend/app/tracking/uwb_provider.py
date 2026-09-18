"""
UWBProvider
-----------
Abstraction pour une source de position physique UWB (section 8 du
cahier des charges "Sûrliv Ball Cup Tracking").

Optionnel par construction : si aucun UWBProvider n'est fourni à
IdentityManager, le pipeline caméra existant (sections 6/9/10/11 de
identity_manager.py) fonctionne EXACTEMENT comme avant. Rien ne change
tant que ce module n'est pas branché explicitement.

Pour brancher votre prototype UWB réel : sous-classez UWBProvider et
implémentez get_position() en lisant votre flux matériel (série, MQTT,
websocket, etc.), puis passez une instance à IdentityManager :

    provider = MyRealUWBProvider(port="/dev/ttyUSB0")
    identity = IdentityManager(tracker, uwb_provider=provider)
"""
from __future__ import annotations
from typing import Optional, Tuple


class UWBProvider:
    """Interface à implémenter pour une vraie source UWB."""

    def get_position(self, frame_index: int) -> Optional[Tuple[float, float, float]]:
        """Retourne (x, y, confidence) en coordonnées image pour cette
        frame, ou None si aucune position UWB n'est disponible pour
        cette frame précise (ne veut pas forcément dire déconnecté)."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """True si le matériel UWB est connecté du tout (section 8 :
        'si le prototype UWB est disponible'), indépendamment d'avoir
        une position pour la frame courante."""
        return False


class NullUWBProvider(UWBProvider):
    """Provider par défaut quand aucun matériel UWB n'est branché :
    no-op garanti, le pipeline caméra reste l'unique source de vérité."""

    def get_position(self, frame_index: int) -> Optional[Tuple[float, float, float]]:
        return None

    def is_available(self) -> bool:
        return False
