network_display = NetworkDisplay.NetworkDisplay() ## have to initialize this here otherwise it gets weird w/ DigitDraw
            network_data = self.compileNetworkData()
            network_display.run(network_data, NUM_TIMESTEPS)