
def sparse_wrapper(env, raw_pnl, period=24):
    # only every 'period' steps the agent receives the accumulated reward
    if env.current_step % period == 0:
        total = sum(env.history[-period:]) + raw_pnl
        return total
    return 0.0
