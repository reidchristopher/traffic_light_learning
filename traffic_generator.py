import numpy as np
import math

# Handle the generation of vehicles in one episode
class TrafficGenerator:

    # Public method
    def generate_routefile(self):

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="can" accel="1.5" decel="4.5" length="5.0"/>""", file=routes)

            print('<flow id="f12" begin="0" end="5400" from="Wi" to="Eo" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f14" begin="0" end="5400" from="Wi" to="No" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f13" begin="0" end="5400" from="Wi" to="So" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f21" begin="0" end="5400" from="Ei" to="Wo" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f23" begin="0" end="5400" from="Ei" to="So" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f24" begin="0" end="5400" from="Ei" to="No" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f34" begin="0" end="5400" from="Si" to="No" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f31" begin="0" end="5400" from="Si" to="Wo" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f32" begin="0" end="5400" from="Si" to="Eo" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f43" begin="0" end="5400" from="Ni" to="So" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f41" begin="0" end="5400" from="Ni" to="Wo" probability="%f"/>' % np.random.uniform(0, 1), file=routes)
            print('<flow id="f42" begin="0" end="5400" from="Ni" to="Eo" probability="%f"/>' % np.random.uniform(0, 1), file=routes)

            print("</routes>", file=routes)

if __name__ == '__main__':
    traffic_generator = TrafficGenerator()
    traffic_generator.generate_routefile()