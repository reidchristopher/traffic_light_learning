import random

# Handle the generation of vehicles in one episode
class TrafficGenerator:

    def __init__(self):
        self.flow_list = ["f12", "f14", "f13", "f21", "f23", "f24", "f34", "f31", "f32", "f43", "f41", "f42"]
        self.from_list = ["W", "E", "S", "N"]
        self.to_list = ["E", "N", "S", "W", "S", "N", "N", "W", "E", "S", "W", "E"]

    # Public method
    def generate_routefile(self, veh_per_hour_list):

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="can" accel="1.5" decel="4.5" length="5.0"/>""", file=routes)

            for i in range(len(self.flow_list)):

                flow = self.flow_list[i]
                from_dir = self.from_list[i // 3]
                to_dir = self.to_list[i]
                prob = veh_per_hour_list[i] / 3600

                print('<flow id="%s" begin="0" from="%si" to="%so" probability="%f"/>' %
                      (flow, from_dir, to_dir, prob), file=routes)

            print("</routes>", file=routes)


if __name__ == '__main__':
    traffic_generator = TrafficGenerator()
    traffic_generator.generate_routefile([250 for _ in range(12)])