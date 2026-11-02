* Cleared RTX temporal accumulation and invalidated cached render updates when cameras
  reset, so the next capture produces fresh images even without advancing simulation
  or requesting additional reset renders.
  The RTX reset affects all cameras, including environment tiles that did not reset.
