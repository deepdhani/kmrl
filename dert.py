from typing import List, Dict

class BrandingManager:
    def __init__(self, contracts: Dict[str, Dict[str, int]]):
        """
        contracts: dictionary of advertiser contracts
        Example:
        {
          "CocaCola": {"target_km": 50000, "current_km": 32000},
          "Airtel": {"target_km": 40000, "current_km": 10000},
          "Tata": {"target_km": 60000, "current_km": 59000}
        }
        """
        self.contracts = contracts

    def calculate_priority_score(self, advertiser: str) -> float:
        """Higher score means more urgent deployment"""
        data = self.contracts[advertiser]
        remaining = data["target_km"] - data["current_km"]
        if remaining <= 0:
            return 0.0  # contract already fulfilled

        progress_ratio = data["current_km"] / data["target_km"]
        urgency_score = 1 - progress_ratio  # closer to 1 = high urgency

        return urgency_score * remaining

    def rank_trainsets(self, trainsets: List[Dict]) -> List[Dict]:
        """
        trainsets: list of dict with 'id' and 'advertiser'
        Example:
        [
          {"id": "Train01", "advertiser": "CocaCola"},
          {"id": "Train02", "advertiser": "Airtel"},
          {"id": "Train03", "advertiser": "Tata"}
        ]
        """
        for t in trainsets:
            adv = t["advertiser"]
            t["branding_priority"] = self.calculate_priority_score(adv)

        # sort descending: highest branding need first
        ranked = sorted(trainsets, key=lambda x: x["branding_priority"], reverse=True)
        return ranked


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    # Branding contracts (targets vs achieved km)
    contracts = {
        "CocaCola": {"target_km": 50000, "current_km": 32000},
        "Airtel": {"target_km": 40000, "current_km": 10000},
        "Tata": {"target_km": 60000, "current_km": 59000}
    }

    manager = BrandingManager(contracts)

    # Trainsets assigned to advertisers
    trainsets = [
        {"id": "Train01", "advertiser": "CocaCola"},
        {"id": "Train02", "advertiser": "Airtel"},
        {"id": "Train03", "advertiser": "Tata"},
    ]

    ranked_list = manager.rank_trainsets(trainsets)

    print("🚆 Branding Priority List (High → Low):")
    for t in ranked_list:
        print(f"{t['id']} - {t['advertiser']} | Priority Score: {t['branding_priority']:.2f}")