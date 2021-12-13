from .planner import Planner, save_model, load_model
import torch
import torchvision.transforms.functional as TF
import numpy as np
from torchvision import transforms
class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        
        self.team = None
        self.num_players = None
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        #self.kart = all_players[np.random.choice(len(all_players))]
        #choose kart
        self.kart = 'tux'
        self.prev_loc = np.int32([0,0])
        self.rescue_count = 0
        self.recovery = False
        self.rescue_steer = 1
        # self.current_team = 'not_sure'
        self.s_turn_left = False
        self.s_turn_right = False
        self.s_count = 0
        self.same_slope = []
        self.face_wall = False
        self.check = False
        self.check2 = False
        #using planner model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model()
        self.model.eval()
        self.goal = 0
        self.facingGoal = True
        self.facingGoal1 = True
        self.backward = False
        self.backward1 = False
        self.frame_count1 = 0
        self.frame_count2 = 0
        self.time_count = 0
        self.reset_count = 0

    def to_numpy(self, location):
        """
        Don't care about location[1], which is the height
        """
        return np.float32([location[0], location[2]])

    def x_intersect(self, kart_loc, kart_front):
        #pos_me,front_me
        #kart_loc => self.to_numpy(player.kart.location)
        #kart_front => self.to_numpy(player.kart.front)
        # print('kart_loc',kart_loc)
        # print('kart_front',kart_front)
        slope = (kart_loc[1] - kart_front[1])/((kart_loc[0] - kart_front[0])+0.000000000000001)
        
        intersect = kart_loc[1] - (slope*kart_loc[0])
        facing_up_grid = kart_front[1] > kart_loc[1]
        if slope == 0:
            x_intersect = kart_loc[1]
        else:
            if facing_up_grid:
                x_intersect = (65-intersect)/slope
            else:
                x_intersect = (-65-intersect)/slope
        return (x_intersect, facing_up_grid)
    

    def model_controller(self, puck_loc, player,player_id,bin):
        """
        This function is called once per timestep. You're given a list of player_states and images.
        DO NOT CALL any pystk functions here. It will crash your program on your grader.
        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D
        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.
        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """


        # distance_down_track
        # finish_time
        # finished_laps
        # front
        # id
        # jumping
        # lap_time
        # lives
        # location
        # max_steer_angle
        # name
        # overall_distance
        # player_id
        # powerup
        # race_result
        # rotation
        # shield_time
        # size
        # velocity
        # wheel_base

        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0, 'fire':False}
        # print(player[0]['kart']['player_id'])
        # print(player[1]['kart']['player_id'])
        # for key, value in player[0]['kart'].items():
        #     print (key)
        #ball_loc = self.to_numpy(state.soccer.ball.location)

        if player_id == 0 or player_id == 1:
            pos_me = self.to_numpy(player[0]['kart']['location'])
            front_me = self.to_numpy(player[0]['kart']['front'])
            kart_velocity = player[0]['kart']['velocity']
        elif player_id == 2 or player_id == 3:
            pos_me = self.to_numpy(player[1]['kart']['location'])
            front_me = self.to_numpy(player[1]['kart']['front'])
            kart_velocity = player[1]['kart']['velocity']

        # velocity_mag = np.sqrt(kart_velocity[0]**2 + kart_velocity[2]**2)
        velocity_mag = np.linalg.norm(kart_velocity)
        x = puck_loc[0]     # 0-400
        y = puck_loc[1]     # 0-300

        if self.team ==0:
            self.team = 'red'
            self.goal = 77
            print('Current Team:',self.team)
        elif self.team ==1:
            self.team = 'blue'
            self.goal = -77
            print('Current Team:',self.team)

        if velocity_mag == 0:
          self.reset_count += 1

        #bug when one kart continue to have velocity zero and keep reset after 50 frame count
        if self.reset_count > 50:
          #reset variable to default
          self.backward = False
          self.backward1 = False
          self.frame_count1 = 0
          self.frame_count2 = 0
          self.time_count = 0
          self.check = False
          self.check2 = False
          if velocity_mag > 0:
            self.reset_count = 0

        if velocity_mag > 20:
            #print('velocity_mag',velocity_mag)
            action['acceleration'] = 0.2

        if (player_id==0 or player_id==1) and velocity_mag > 5 and self.check == False:
          self.check = True

        if (player_id==2 or player_id==3) and velocity_mag >  5 and self.check2 == False:
          self.check2 = True
        
        if player_id == 0 or player_id == 1:
          #check if facing opponent goal
          if self.team == 'red':
            if front_me[1] < self.goal:
              facingGoal = True
            else:
              facingGoal = False
          else:
            if front_me[1] > self.goal:
              facingGoal = True
            else:
              facingGoal = False
 
          #back condition check
          if (-45 < pos_me[0] < 45 ) and (-64 < pos_me[1] < 64):
            if self.check == True and velocity_mag < 0.2:
              if pos_me[1] > 63.5:
                self.backward = True
                self.frame_count1 = 10
              elif pos_me[1] <-63.5:
                if self.team == 'blue':
                  self.backward = True
                  self.frame_count1 = 10
                else:
                  action['rescue'] = True
              else:
                  action['rescue'] = True
              self.check = False
            if self.frame_count1 == 0:
              self.backward = False
          else: #inside a goal check if backward or forward
            if (pos_me[1] > 64 and front_me[1]>64) or (pos_me[1] < -64 and front_me[1] < -64):
              if (front_me[1] - pos_me[1]> 0) and pos_me[1] < -64:
                self.backward2 = False
              elif (front_me[1] - pos_me[1] < 0) and pos_me[1] > 64:
                self.backward2 = False
              elif self.frame_count1 == 0:
                self.backward = True
                self.frame_count1 = 20
            else:
              if self.check == True and velocity_mag < 0.2:
                action['rescue'] = True
                self.check = False
              else:
                self.backward = False
                self.frame_count1 == 0
         
          #backwarding
          if self.backward == True and self.frame_count1 > 0:
            action['brake'] = True
            action['acceleration'] = 0
            if pos_me[1] < -64 or pos_me[1]>64:
              if pos_me[1] < -64:
                if self.check == True and self.team == 'red':
                  self.check = False
                  self.backward = False
                  self.frame_count1 == 0
                  action['rescue'] = True
                else:
                  if front_me[1] - pos_me[1] > 0:
                    self.backward = False
                    self.frame_count1 = 0
                  else: #backward direction
                    if pos_me[0] < -5:
                      action['steer'] = -1
                    elif pos_me[0] > 5:
                      action['steer'] = 1
                    else:
                      action['steer'] = 0
              if pos_me[1] > 64:
                """
                need to fix when blue kart enter own goal
                """
                if self.check == True and self.team == 'blue':
                  self.check = False
                  self.backward = False
                  self.frame_count1 == 0
                  action['rescue'] = True
                else:
                  if front_me[1] - pos_me[1] < 0:
                    self.backward = False
                    self.frame_count1 = 0
                  else:
                      action['steer'] = 0
            elif x < 62:
              action['steer'] = 1
            elif x > 66:
              action['steer'] = -1
            self.frame_count1 -= 1
          else:
            self.backward = False
            self.frame_count1 = 0

          if self.backward == False:
            if front_me[0] < -10:
              if facingGoal == True:
                action['nitro'] = True
                if x < 62:
                  action['steer'] = -1
                elif x > 66:
                  action['steer'] = 1
                else:
                  if y > 48:
                    action['acceleration'] = 0.25
                  elif 35 < y < 48:
                    action['acceleration'] = 0.75
                  action['fire'] = True
                  action['steer'] = 0
              else:
                action['nitro'] = False
                if x < 62:
                  action['steer'] = -1
                elif x > 66:
                  action['steer'] = 0.75
                else:
                  action['acceleration'] = 0.75
                  action['fire'] = False
                  action['steer'] = -0.15
          #right side
            elif front_me[0] > 10:
              if facingGoal == True:
                action['nitro'] = True
                if x < 62:
                  action['steer'] = -1
                elif x > 66:
                  action['steer'] = 1
                else:
                  if y > 48:
                    action['acceleration'] = 0.25
                  elif 35 < y < 48:
                    action['acceleration'] = 0.75
                  action['fire'] = True
                  action['steer'] = 0
              else:
                action['nitro'] = False
                if x < 62.5:
                  action['steer'] = -0.75
                elif x > 65.5:
                  action['steer'] = 1
                else:
                  action['acceleration'] = 0.75
                  action['fire'] = False
                  action['steer'] = 0.15
            #center
            else:
              if facingGoal == True:
                #if inside goal when not backwarding accelerate forward
                if pos_me[1] < -64:
                  if pos_me[0] < 0:
                    action['steer'] = -1
                  else:
                    action['steer'] = 1
                elif pos_me[1] > 64:
                  if pos_me[0] < 0:
                    action['steer'] = 1
                  else:
                    action['steer'] = -1
                else:
                  action['nitro'] = True
                  if self.time_count < 10:
                    action['acceleration'] = 1
                    action['steer'] = 0
                  else:
                    if x < 62:
                      action['steer'] = -1
                    elif x > 66:
                      action['steer'] = 1
                    else:
                      if y > 48:
                        action['acceleration'] = 0.25
                      elif 35 < y < 48:
                        action['acceleration'] = 0.75
                      action['fire'] = True
                      action['steer'] = 0
              else:
                action['nitro'] = False
                if x < 56:
                  action['steer'] = -0.75
                elif x > 72:
                  action['steer'] = 0.75
                else:
                  action['acceleration'] = 0.75
                  action['fire'] = False
                  action['steer'] = 0
            
        """
        anything fix above will also need to fix below
        """
        if player_id == 2 or player_id == 3:
          if self.backward1 == True and self.frame_count2 > 0:
            action['brake'] = True
            action['acceleration'] = 0
            if pos_me[1] < -64 or pos_me[1]>64:
              if pos_me[1] < -64: #inside goal
                if self.team == 'red' and self.check2 == True:
                  self.backward1 = False
                  self.frame_count2 == 0
                  action['rescue'] = True
                  self.check2 = False
                  #rescue when red in red team
                else:
                  if front_me[1] - pos_me[1] > 0:
                    self.backward1 = False
                    self.frame_count2 == 0
                  else:
                      action['steer'] = 0
              if pos_me[1]>64:
                if self.team == 'blue' and self.check2 == True:
                  self.backward1 = False
                  self.frame_count2 == 0
                  action['rescue'] = True
                  self.check2 = False
                  #rescue when blue team
                else:
                  if front_me[1] - pos_me[1] < 0:
                    self.backward1 = False
                    self.frame_count2 == 0
                  else:
                      action['steer'] = 0
            elif x < 62:
              action['steer'] = 1
            elif x > 66:
              action['steer'] = -1
            self.frame_count2 -= 1
          else:
            self.backward1 = False
            self.frame_count2 = 0
          #backward
          if (-45 < pos_me[0] < 45 ) and (-64 < pos_me[1] < 64):
            if self.check2 == True and velocity_mag < 0.2:
              if pos_me[1] > 63.5:
                self.backward1 = True
                self.frame_count2 = 10
              elif pos_me[1] <-63.5:
                if self.team == 'blue':
                  self.backward1 = True
                  self.frame_count2 = 10
                else:
                  action['rescue'] = True
              else:
                  action['rescue'] = True
              self.check2 = False
            if self.frame_count2 == 0:
              self.backward1 = False
          else:
            if (pos_me[1] > 64 and front_me[1]>64) or (pos_me[1] < -64 and front_me[1] < -64):
              if (front_me[1] - pos_me[1]> 0) and pos_me[1] < -64:
                self.backward2 = False
              elif (front_me[1] - pos_me[1] < 0) and pos_me[1] > 64:
                self.backward2 = False
              elif self.frame_count2 == 0:
                self.backward1 = True
                self.frame_count2 = 20
            else:
              if self.check2 == True and velocity_mag < 0.2:
                action['rescue'] = True
                self.check2 = False
              else:
                self.backward1 = False
                self.frame_count2 == 0

          #facing goal check
          if self.team == 'red':
            if front_me[1] < self.goal:
              facingGoal1 = True
            else:
              facingGoal1 = False
          else:
            if front_me[1] > self.goal:
              facingGoal1 = True
            else:
              facingGoal1 = False

          if self.backward1 == False:
            #left side
            if front_me[0] < -10:
              if facingGoal1 == True:
                action['nitro'] = True
                if x < 62:
                  action['steer'] = -1
                elif x > 66:
                  action['steer'] = 1
                else:
                  if y > 48:
                    action['acceleration'] = 0.25
                  elif 35 < y < 48:
                    action['acceleration'] = 0.75
                  action['fire'] = True
                  action['steer'] = 0
              else:
                action['nitro'] = False
                if x < 62.:
                  action['steer'] = -1
                elif x > 66:
                  action['steer'] = 0.75
                else:
                  action['acceleration'] = 0.75
                  action['fire'] = False
                  action['steer'] = -0.15
            #right side
            elif front_me[0] > 10:
              if facingGoal1 == True:
                action['nitro'] = True
                if x < 62:
                  action['steer'] = -1
                elif x > 66:
                  action['steer'] = 1
                else:
                  if y > 48:
                    action['acceleration'] = 0.25
                  elif 35 < y < 48:
                    action['acceleration'] = 0.75
                  action['fire'] = True
                  action['steer'] = 0
              else:
                action['nitro'] = False
                if x < 62:
                  action['steer'] = -0.75
                elif x > 66:
                  action['steer'] = 1
                else:
                  action['acceleration'] = 0.75
                  action['fire'] = False
                  action['steer'] = 0.15
            #center
            else:
              if facingGoal1 == True:
                if pos_me[1] < -64:
                  if pos_me[0] < 0:
                    action['steer'] = -1
                  else:
                    action['steer'] = 1
                elif pos_me[1] > 64:
                  if pos_me[0] < 0:
                    action['steer'] = 1
                  else:
                    action['steer'] = -1
                else:
                  action['nitro'] = True
                  if self.time_count < 10:
                    action['acceleration'] = 1
                    action['steer'] = 0
                  else:

                    if x < 62:
                      action['steer'] = -1
                    elif x > 66:
                      action['steer'] = 1
                    else:
                      action['acceleration'] = 0.25
                      action['fire'] = True
                      action['steer'] = 0
              else:
                action['nitro'] = False
                if x < 56:
                  action['steer'] = -0.75
                elif x > 72:
                  action['steer'] = 0.75
                else:
                  action['acceleration'] = 0.75
                  action['fire'] = False
                  action['steer'] = 0

        if x < 16 or x > 112:
            action['drift'] = True
            action['acceleration'] = 0.2
        else:
            action['drift'] = False

        if x < 32 or x > 96:
            action['acceleration'] = 0.5

        return action
    
    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['nolok','pidgin']

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.
        DO NOT CALL any pystk functions here. It will crash your program on your grader.
        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D
        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.
        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        # distance_down_track
        # finish_time
        # finished_laps
        # front
        # id
        # jumping
        # lap_time
        # lives
        # location
        # max_steer_angle
        # name
        # overall_distance
        # player_id
        # powerup
        # race_result
        # rotation
        # shield_time
        # size
        # velocity
        # wheel_base

      
        self.time_count += 1
        player_id1 = player_state[0]['kart']['id']
        player_id2 = player_state[1]['kart']['id']
        # print('player_id1',player_id1)
        # print('player_id2',player_id2)
        # print(TF.to_tensor(player_image[0]).shape)
        im1 = transforms.ToPILImage()(TF.to_tensor(player_image[0])[None].squeeze(0)).convert("RGB")
        im2 = transforms.ToPILImage()(TF.to_tensor(player_image[1])[None].squeeze(0)).convert("RGB")
        resize = transforms.Resize([96,128])
        im1 = resize(im1)
        im2 = resize(im2)
        pil_to_tensor1 = transforms.ToTensor()(im1)
        pil_to_tensor2 = transforms.ToTensor()(im2)
        # print(type(im))
        '''
        model_puck_loc = self.model(pil_to_tensor1[None]).squeeze(0).cpu().detach().numpy()
        model_puck_loc2 = self.model(pil_to_tensor2[None]).squeeze(0).cpu().detach().numpy()
        # print(model_puck_loc)
        '''
        
        model_puck_loc, bin = self.model(pil_to_tensor1[None])
        model_puck_loc2, bin2 = self.model(pil_to_tensor2[None])
        model_puck_loc = model_puck_loc.squeeze(0).cpu().detach().numpy()   # + 1) * [64, 48]    # Only doing (x + 1) * [64, 48] when working with normalized image coordinates. 
        model_puck_loc2 = model_puck_loc2.squeeze(0).cpu().detach().numpy() #+ 1) * [64, 48]  # Only doing (x + 1) * [64, 48] when working with normalized image coordinates. 
        #bin = bin.cpu().detach().numpy()
        #bin2 = bin2.cpu().detach().numpy()

        model_action = self.model_controller(model_puck_loc, player_state,player_id1,bin)
        model_action2 = self.model_controller(model_puck_loc2, player_state,player_id2,bin2)

        #print(model_puck_loc)
        #print(model_action)
        return [model_action,model_action2]