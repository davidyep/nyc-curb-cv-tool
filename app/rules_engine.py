from __future__ import annotations

from pathlib import Path

import yaml

from app.schemas import FrameContext, LegalityDecision, VehicleObservation, ZoneDefinition


class RulesEngine:
    def __init__(self, rules_path: str = "config/nyc_parking_rules.yaml") -> None:
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        if not self.rules_path.exists():
            return {}
        with self.rules_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def evaluate(self, frame: FrameContext, observation: VehicleObservation) -> LegalityDecision:
        reason_codes: list[str] = []
        confidence = 0.9

        dwell_limit = self.rules.get("dwell_time_limits", {}).get(observation.vehicle_type, 900)

        if observation.is_double_parked:
            reason_codes.append("double_parking_detected")
        if observation.is_obstructing:
            reason_codes.append("critical_obstruction")
        if observation.lane_type in ("bus", "bike"):
            reason_codes.append(f"{observation.lane_type}_lane_occupied")
        if observation.dwell_time_seconds > dwell_limit:
            reason_codes.append("dwell_time_exceeded")

        overnight = frame.timestamp_utc.hour >= 0 and frame.timestamp_utc.hour < 6
        if overnight and observation.vehicle_type == "commercial":
            reason_codes.append("overnight_commercial_restriction")

        if not reason_codes:
            status = "legal"
            confidence = 0.95
        elif "critical_obstruction" in reason_codes or "double_parking_detected" in reason_codes:
            status = "likely_illegal"
            confidence = 0.92
        elif any(code.endswith("lane_occupied") for code in reason_codes):
            status = "likely_illegal"
            confidence = 0.88
        else:
            status = "uncertain"
            confidence = 0.75

        return LegalityDecision(
            track_id=observation.track_id,
            status=status,
            reason_codes=reason_codes,
            confidence=confidence,
        )

    def evaluate_with_zone(
        self,
        frame: FrameContext,
        observation: VehicleObservation,
        zone: ZoneDefinition | None,
    ) -> LegalityDecision:
        """Extended evaluation that layers zone-specific rules on top of base checks."""
        decision = self.evaluate(frame, observation)
        reason_codes = list(decision.reason_codes)

        if zone is None:
            return decision

        # Zone-specific rules
        if zone.zone_type == "no_parking":
            reason_codes.append("no_parking_zone_violation")
        if zone.zone_type == "fire_hydrant":
            reason_codes.append("fire_hydrant_zone_violation")
        if zone.zone_type == "travel_lane":
            reason_codes.append("travel_lane_violation")
        if zone.zone_type == "bus_lane" and observation.vehicle_type != "bus":
            if "bus_lane_occupied" not in reason_codes:
                reason_codes.append("bus_lane_occupied")
        if zone.zone_type == "bike_lane" and observation.vehicle_type != "bike":
            if "bike_lane_occupied" not in reason_codes:
                reason_codes.append("bike_lane_occupied")
        if zone.zone_type == "loading_zone" and observation.vehicle_type == "passenger":
            dwell_limit = self.rules.get("dwell_time_limits", {}).get("passenger", 900)
            if observation.dwell_time_seconds > dwell_limit:
                reason_codes.append("loading_zone_passenger_overstay")

        # Re-derive status from full set of reason codes
        if not reason_codes:
            status: str = "legal"
            confidence = 0.95
        elif any(
            code in reason_codes
            for code in (
                "fire_hydrant_zone_violation",
                "no_parking_zone_violation",
                "travel_lane_violation",
                "double_parking_detected",
                "critical_obstruction",
            )
        ):
            status = "likely_illegal"
            confidence = 0.92
        elif any(code.endswith("_occupied") for code in reason_codes):
            status = "likely_illegal"
            confidence = 0.88
        else:
            status = "uncertain"
            confidence = 0.75

        return LegalityDecision(
            track_id=observation.track_id,
            status=status,
            reason_codes=reason_codes,
            confidence=confidence,
        )
