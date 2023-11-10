% Setting the resolution of the video frame
x_res = 640;
y_res = 480;

% Calculate the middle of the frame
frame_middle = [x_res/2, y_res/2];

% Define a bounding box for face detection
face_detect_bbox_rectangle = [x_res-2, y_res-2, x_res-2, y_res-2];

% Reshape the bounding box coordinates
face_detect_bbox = reshape(bbox2points(face_detect_bbox_rectangle)', 1, []);

% Create a face detector object
faceDetector = vision.CascadeObjectDetector('MinSize', [floor(x_res/6), floor(y_res/6)]);

% Initialize the webcam
webcamlist()
cam = webcam(1);

% Capture one frame to get its size
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create a video player to display frames
videoPlayer = vision.VideoPlayer('Position', [200 200 [frameSize(2), frameSize(1)] + 30]);

% Initialize the robot
robot = MyRobot();
assert(robot.is_robot_connected(), "Robot not connected properly");
robot.move_j(0, -90, 0, 0);
cw = 0;
pause(2);

try
    % Capture a frame from the webcam
    videoFrame = snapshot(cam);
    bbox = faceDetector.step(videoFrame);

    if ~isempty(bbox)
        % Get the bounding box points of the detected face
        bboxPoints = bbox2points(bbox(1, :));
        bboxPolygon = reshape(bboxPoints', 1, []);
        
        % Calculate the center of the detected face
        center = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];

        % Visualize face detection on the video frame
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3, 'Color', "blue");
        videoFrame = insertShape(videoFrame, 'Circle', [frame_middle, 5], 'LineWidth', 5, 'Color', "red");
        videoFrame = insertShape(videoFrame, 'Line', [frame_middle, center], 'LineWidth', 5, 'Color', "red");

        % Calculate deltas for robot movement
        delta_j1 = (frame_middle(1) - center(1)) / 30;
        delta_j4 = (frame_middle(2) - center(2)) / 30;

        try
            % Move the robot joints based on the deltas
            robot.move_j(robot.joint_angles(1) + delta_j1, robot.joint_angles(2), robot.joint_angles(3), robot.joint_angles(4) + delta_j4);
        catch ME
            disp(ME.message);
        end
    end

    % Display the annotated video frame using the video player
    step(videoPlayer, videoFrame);

    % Check if the video player window is closed
    runLoop = isOpen(videoPlayer);
    if ~runLoop
        % Clean up resources
        release(videoPlayer);
        release(faceDetector);
        robot.disable_motors();
    end
end

% Helper function to jog the robot in the Z-axis
function cw = jog_robot_z(cw, robot)
    % Check joint angles for robot limits
    if robot.joint_angles(1) == -130
        cw = 1;
    elseif robot.joint_angles(1) == 130
        cw = 0;
    end
    
    % Move the robot based on cw (clockwise) flag
    if ~cw
       robot.move_j(robot.joint_angles(1) - 10, robot.joint_angles(2), robot.joint_angles(3), robot.joint_angles(4));
    else
        robot.move_j(robot.joint_angles(1) + 10, robot.joint_angles(2), robot.joint_angles(3), robot.joint_angles(4));
    end
end

% Helper function to calculate distances between points
function dists = get_distances(center, frame_middle)
    x_dist = frame_middle(1) - center(1);
    y_dist = frame_middle(2) - center(2);
    dists = [x_dist, y_dist];
end

% Helper function to get the center of a bounding box polygon
function center = get_center(bboxPolygon)
    x = floor((bboxPolygon(1) + bboxPolygon(3)) / 2);
    y = floor((bboxPolygon(2) + bboxPolygon(4)) / 2);
    center = [x, y];
end

% Helper function to check if a face is inside the bounding box
function inside = face_inside_bbox(center, frame_middle)
    if abs(frame_middle(1) - center(1)) < frame_middle(1) && abs(frame_middle(2) - center(2)) < frame_middle(2)
        inside = 1;
    else
        inside = 0;
    end
end
