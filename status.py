#!/usr/bin/env python3

import light_manager

lm = light_manager.LightManager()
lm.on(7, False)
print(lm.status)
