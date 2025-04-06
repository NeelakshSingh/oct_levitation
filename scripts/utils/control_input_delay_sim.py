import rospy
from collections import deque
from tnb_mns_driver.msg import DesCurrentsReg

class DelayedPublisher:
    def __init__(self):
        rospy.init_node('delayed_publisher')
        self.delay_nsec = 1e9*rospy.get_param("free_body_sim_utils/delay_time", 0.01)
        self.delay_republisher_freq = rospy.get_param("free_body_sim_utils/delay_repub_freq", 500)
        self.message_queue = deque()
        rospy.loginfo(f"[Control Delay Sim] Publishing delayed currents with a delay of {self.delay_nsec/1e9} sec.")
        self.pub = rospy.Publisher('/tnb_mns_driver/des_currents_reg/delayed_sim', DesCurrentsReg, queue_size=100)
        self.sub = rospy.Subscriber('/tnb_mns_driver/des_currents_reg', DesCurrentsReg, self.callback, queue_size=100)

    def callback(self, msg):
        self.message_queue.append((rospy.Time.now().to_nsec(), msg))

    def publish_delayed_messages(self, event):
        current_time = rospy.Time.now().to_nsec()
        # Check if there are any messages that should be published
        if self.message_queue and (current_time - self.message_queue[0][0]) >= self.delay_nsec:
            _, msg = self.message_queue.popleft()
            self.pub.publish(msg)
    
    def run(self):
        self.repub_timer = rospy.Timer(rospy.Duration(1/self.delay_republisher_freq), self.publish_delayed_messages)
        rospy.spin()

if __name__ == '__main__':
    delayed_publisher = DelayedPublisher()
    delayed_publisher.run()