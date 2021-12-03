from .planner import Planner, save_model, load_model
import torch
import torchvision.transforms.functional as TF
import numpy as np
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
        
        #using planner model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model()
        self.model.eval()
    

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
    

    def model_controller(self, puck_loc, player,player_id):
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

        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
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

        #abs_x = np.abs(x)
        sign_x = np.sign(x)

        # clipping x and y values
        if x < 0:
            x=0
        if x > 400:
            x=400

        if y < 0:
            y=0
        if y > 300:
            y=300
        if self.team ==0:
            self.team = 'red'
            print('Current Team:',self.team)
        elif self.team ==1:
            self.team = 'blue'
            print('Current Team:',self.team)

        #print('kart id',player.kart.id,'velocity',player.kart.velocity)

        # LEAN FEATURE
        
        x_intersect, facing_up_grid = self.x_intersect(pos_me,front_me)
        
        lean_val = 2
 

        #if (175<x<225) and (100<y<120):
            #lean_val = 10
        #if (175<x<225) and (120<y<160):
            #lean_val = 5

        if -10<pos_me[0]<10:
            lean_val=0

        # facing outside goal
        #if facing_up_grid and 9<x_intersect<40 and pos_me[0]>9:
        #if facing_up_grid:
            #print('facing_up_grid',facing_up_grid)
        #print('x_intersect',x_intersect)
        #print('y loc',pos_me[1])
        #if self.current_team == 'red':
        if facing_up_grid and 9<x_intersect<40:
            #if red team
        
            if self.team == 'red':
                
                x += lean_val
                #print('RL_F')
            else:
                x -= lean_val
                #print('RL_B')
            #if (150<x<250) and velocity_mag > 12:
                #action['acceleration'] = 0.5
                #action['brake'] = True
                #print('RL_F')
        #if facing_up_grid and -40<x_intersect<-9 and pos_me[0]<-9:
        
        if facing_up_grid and -40<x_intersect<-9:
            #if red team
  
            if self.team == 'red':
                x -= lean_val
                #print('LR_F')
            else:
                x += lean_val
                #print('LR_B')
            #if (150<x<250) and velocity_mag > 12:
                #action['acceleration'] = 0.5
                #action['brake'] = True
                #print('LR_F')

        # facing inside goal
        if (not facing_up_grid) and 0<x_intersect<10:
            #if red team
   
            if self.team == 'red':
                x += lean_val
                #print('RL_B')
            else:
                x -= lean_val
                #print('RL_F')
        if (not facing_up_grid) and -10<x_intersect<0:
            #if red team
      
            if self.team == 'red':
                x -= lean_val
                #print('LR_B')
            else:
                x += lean_val
                #print('LR_F')
            
   
        #print('velocity_mag',velocity_mag)
        if velocity_mag > 20:
            #print('velocity_mag',velocity_mag)
            action['acceleration'] = 0.2

        
        if x < 200:
            action['steer'] = -1
        elif x > 200:
            action['steer'] = 1
        else:
            action['steer'] = 0
            # here you can put shot alignment, or narrow edging depending on where kart is and where it is facing

        if x < 50 or x > 350:
            action['drift'] = True
            action['acceleration'] = 0.2
        else:
            action['drift'] = False

        if x < 100 or x > 300:
            action['acceleration'] = 0.5

  
        # RECOVERY 

        if self.recovery == True:
            action['steer'] = self.rescue_steer
            action['acceleration'] = 0
            action['brake'] = True
            self.rescue_count -= 2
            #print('rescue_count',self.rescue_count)
            # no rescue if initial condition
            if self.rescue_count < 1 or ((-57<pos_me[1]<57 and -7<pos_me[0]<1) and velocity_mag < 5):
                self.rescue_count = 0
                self.recovery = False
        else:
            if self.prev_loc[0] == np.int32(pos_me)[0] and self.prev_loc[1] == np.int32(pos_me)[1]:
                self.rescue_count += 5
            else:
                if self.recovery == False:
                    self.rescue_count = 0

            #if self.rescue_count > 30 or pos_me[1]>65 or pos_me[1]<-65:
            #if self.rescue_count > 30 or ((y>200) and (50<x<350)):
            if self.rescue_count<2:
                if x<200:
                    self.rescue_steer = 1
                else:
                    self.rescue_steer = -1
            if self.rescue_count > 30 or (y>200):
                #if x<200:
                    #self.rescue_steer = 1
                #else:
                    #self.rescue_steer = -1
                # case of puck near bottom left/right
                if velocity_mag > 10:
                    self.rescue_count = 30
                    self.rescue_steer = 0
                else:
                    self.rescue_count = 20
                self.recovery = True

        self.prev_loc = np.int32(pos_me)
   
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
        player_id1 = player_state[0]['kart']['id']
        player_id2 = player_state[1]['kart']['id']
        # print('player_id1',player_id1)
        # print('player_id2',player_id2)
        model_puck_loc = self.model(TF.to_tensor(player_image[0])[None]).squeeze(0).cpu().detach().numpy()
        model_puck_loc2 = self.model(TF.to_tensor(player_image[1])[None]).squeeze(0).cpu().detach().numpy()

        model_action = self.model_controller(model_puck_loc, player_state,player_id1)
        model_action2 = self.model_controller(model_puck_loc2, player_state,player_id2)
        return [model_action,model_action2]
