class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.agent_costs = []

    def add(self, agent: str, input_tokens: int, output_tokens: int, cost: float):
        """Add cost entry for an agent."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.agent_costs.append({
            "agent": agent,
            "tokens": input_tokens + output_tokens,
            "cost_usd": round(cost, 4)
        })

    def report(self):
        """Get cost summary report."""
        return {
            "total_usd": round(self.total_cost, 4),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "agents_used": self.agent_costs
        }
