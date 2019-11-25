import random

# Handle the generation of vehicles in one episode
class TrafficGenerator:

    def __init__(self, min_number=1000, max_number=3100, step=500):
        self.min_number = min_number
        self.max_number = max_number
        self.step = step

    # Public method
    def generate_routefile(self):
        
        vehsPerHour = random.randrange(self.min_number, self.max_number, self.step)
        print(vehsPerHour)
        vehsPerHour_straight = int((vehsPerHour*0.5) / 4) # The number of vehicles going straight for each lane per hour
        vehsPerHour_turning = int((vehsPerHour*0.5) / 8) # The number of vehicles turning for each lane pe hour

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="can" accel="1.5" decel="4.5" length="5.0"/>""", file=routes)

            print('<flow id="f12" begin="0" end="5400" from="Wi" to="Eo" vehsPerHour="%f"/>' % vehsPerHour_straight, file=routes)
            print('<flow id="f14" begin="0" end="5400" from="Wi" to="No" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f13" begin="0" end="5400" from="Wi" to="So" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f21" begin="0" end="5400" from="Ei" to="Wo" vehsPerHour="%f"/>' % vehsPerHour_straight, file=routes)
            print('<flow id="f23" begin="0" end="5400" from="Ei" to="So" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f24" begin="0" end="5400" from="Ei" to="No" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f34" begin="0" end="5400" from="Si" to="No" vehsPerHour="%f"/>' % vehsPerHour_straight, file=routes)
            print('<flow id="f31" begin="0" end="5400" from="Si" to="Wo" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f32" begin="0" end="5400" from="Si" to="Eo" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f43" begin="0" end="5400" from="Ni" to="So" vehsPerHour="%f"/>' % vehsPerHour_straight, file=routes)
            print('<flow id="f41" begin="0" end="5400" from="Ni" to="Wo" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)
            print('<flow id="f42" begin="0" end="5400" from="Ni" to="Eo" vehsPerHour="%f"/>' % vehsPerHour_turning, file=routes)

            print("</routes>", file=routes)

if __name__ == '__main__':
    traffic_generator = TrafficGenerator()
    traffic_generator.generate_routefile()