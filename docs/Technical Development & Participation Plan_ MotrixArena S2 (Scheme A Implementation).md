### Technical Development & Participation Plan: MotrixArena S2 (Scheme A Implementation)

#### 1\. Executive Strategic Context

The MotrixArena S2 challenge represents the premier "Experimental Field for Embodied AI," providing a rigorous environment to transition abstract Reinforcement Learning (RL) algorithms into a high-fidelity 3v3 simulated soccer arena. This competition is a comprehensive stress test for autonomous control systems, requiring precise orchestration of motor skills under adversarial pressure. For the Senior Robotics Architect, this is not merely a game but a validation of the bridge between low-level locomotion and high-level tactical coordination.The engineering challenge is codified into four critical pillars that our architecture must address:

* **Stable Walking:**  Ensuring robust bipedal locomotion on a dynamic pitch.  
* **Multi-body Coordination:**  Synchronizing three units to maintain tactical integrity.  
* **Real-time Decision Making:**  Processing environment states to issue low-latency commands.  
* **Adversarial Games:**  Adapting to unpredictable opponent behaviors in real-time.Scheme A (K1 AMP) serves as our foundational solution. By leveraging a Full Joint Gait framework, we provide the necessary physical stability to support complex tactical maneuvers. Technical success is predicated on rigorous adherence to the platform's architectural constraints; non-compliance with the simulation's loading logic constitutes an immediate failure point.

#### 2\. Core Technical Architecture: Scheme A (K1 AMP) Specifications

Adopting Scheme A (K1 AMP / Full Joint Gait) is a strategic requirement to align with the native MotrixSim loading logic and the high-performance capabilities of the  **Booster K1 Geek Edition**  hardware. This alignment ensures that the gait model can fully utilize the physics engine's PD control gains for the 3v3 humanoid matches.

##### Technical Specification Table: Scheme A (K1 AMP)

Feature,Specification,Engineering Mandate  
Interface Shape,"N, 375 → N, 22",Inference Batching:  Typically export  N=1 . The simulation handles batching across robots internally.  
Observation Logic,5-frame stack (75 dims × 5),Mandatory 375-dimension input for state history.  
Observation Semantics,"Command, Gravity, Gyro, Joint Position Error, Joint Velocity, Previous Actions",Must follow K1\_amp\_config definitions in the training repo.  
Action Mapping,22-dimensional output,Must align precisely with K1\_AMP\_ACTUATOR\_JOINT\_ORDER and MJCF motor sequences.  
Control Format,.onnx (Recommended) or .pt,TorchScript is a valid fallback; ONNX is preferred for inference speed.  
Gait Switch,--k1-legged-gym,Must be enabled to prevent mixing with full-body AMP dimensions.

##### Architectural Insight: Scaling and Stability

The engineering team shall enforce the K1\_AMP\_ACTION\_SCALE \= 0.25 constant. This is not merely a limiter; it is a critical alignment factor for the physics engine's PD controller. A scale of 0.25 ensures that the output motor sequences stay within the MJCF motor limits, preventing actuator saturation and maintaining PD control stability during high-velocity transitions. Failure to apply this scale correctly will result in "posture collapse" or erratic oscillating behaviors. These motor skills are the execution arm for the Decider layer's movement vectors.

#### 3\. Decider-Gait Integration & Communication Protocol

Our architecture maintains a strict "Separation of Concerns." The Gait model handles the "How" (locomotion stability), while the Decider handles the "Where" (tactical positioning and ball interaction).

##### ZMQ Communication Layer

The Decider processes environmental states and issues velocity instructions (cmd) via ZMQ. The following JSON structure is mandatory for all team agents:  
{  
  "id": 0, // Red Team: 0-6; Blue Team: 7-13  
  "cmd": \[0.5, 0.1, 0.05\] // \[vx, vy, w\] normalized to robot coordinate system  
}

##### Protocol Compliance & Robustness

Engineering leads must account for the following communication risks:

1. **ID Allocation:**  Ensure Red Team agents use the 0-6 range and Blue Team use 7-13 to prevent cross-team command leakage.  
2. **Normalization & Dead Zones:**  Commands are subject to CMD\_VEL\_NORM clipping. The Decider must account for "Dead Zones" where low-magnitude commands (too small) will not trigger gait movement, and high-magnitude commands will be clipped by simulation logic (Section 4.2).  
3. **Written Confirmation:**  Any modifications to the ZMQ packet structure or JSON schema require written confirmation between the Strategy and Integration teams to prevent simulation-side parsing errors.

#### 4\. Environment Standardization & Deployment Readiness

To eliminate "works on my machine" integration failures, all developers must synchronize with the official MotrixArena S2 environment.

##### Deployment Requirements

* **Conda Environment:**  All inference and simulation tasks shall run within the motrixsim0508 environment.  
* **Decider Dependency Locking:**  All libraries must be frozen in decider/requirements.txt using pip freeze. Python version locking is non-negotiable.  
* **Inference Engine:**  onnxruntime is mandatory on the simulation side. When exporting models, the specific opset version must be documented in the README\_Gait.md.  
* **Hardware Dependencies:**  If Decider logic utilizes GPU acceleration, the exact CUDA/cuDNN versions must be specified. Simulation ONNX inference defaults to CPU for maximum stability.  
* **Safe Paths Policy:**  All model and code paths must strictly forbid non-standard characters (Chinese characters, spaces). Non-compliance will cause path-resolution failures on the simulation server.

#### 5\. Quality Assurance & Self-Verification Protocol

The "Simulation Acceptance Criteria" are the final hurdle before submission. Every model must pass the following verification pipeline.

##### Pre-Submission Checklist

1. **Dimension Verification:**  Use onnxruntime to confirm single-input/single-output shapes of N, 375 and N, 22\.  
2. **Runtime Integrity:**  The Decider must demonstrate continuous state reception and cmd issuance without memory leaks or stack overflows over a 20-minute test loop.  
3. **Physical Verification:**  Use the \--policy flag to point to the model's absolute path and confirm the robot can maintain posture/walk using the \--k1-legged-gym switch.  
4. **Autonomous Fall-Recovery Test:**  Force front, back, and side falls and confirm that the submitted policy begins a stand-up attempt within 20 seconds, detects success or failure, limits failure handling to 3 consecutive attempts, and resumes locomotion after standing. Simulator-side teleportation or unconditional pose reset must be disabled for this test.
5. **Penalty and Return Test:**  Confirm that an Incapable Robot is removed for an initial 30-second penalty, that repeated violations accumulate penalty time, and that the released robot autonomously returns to the field.
6. **The Clean Machine Test:**  Reproduce the entire setup on a secondary, isolated machine using only the provided documentation and pip install \-r requirements.txt. If the setup is not reproducible in one step, it is rejected.

#### 6\. Competition Roadmap & Milestone Management

Strategic success requires temporal alignment with the MotrixArena S2 official broadcast schedule. We are preparing for a high-volume competition featuring 48 total matches in the Round Robin alone.

##### Competition Schedule (2026)

Stage,Date Range,Core Tasks & Volume  
Registration,May 1 – May 15,Team registration and intent filing.  
Model Submission,May 15 – June 5,"Finalization of RL models, .onnx exports, and README\_Gait.md."  
Qualifying (32-Team),June 8 – June 12,Initial performance screening and seed selection.  
Round Robin,June 12 – June 24,12 days of live broadcasts; 4 matches per day (48 total).  
Elimination Bracket,June 27 – July 3,"16-team through Semifinals; weekend ""Golden Time"" broadcasts."  
Grand Finals,July 5,"Finals, All-Star performance, and award ceremony."

##### Tactical Response: The Penalty Shootout

In the event of a draw, matches enter a "10-minute high-frequency burst" point-blank shootout. This requires the Gait controller to possess "Accumulated Error Reset" logic to prevent physical collapse during repetitive, high-intensity kicks. Our robustness strategy must prioritize posture recovery over aggressive movement during this phase.

##### Final Delivery & 3-Dimensional Assessment

All submissions must include a clear README\_Gait.md and exclude private tokens, logs, or /opt dependencies. Our final performance will be evaluated against the  **3-Dimensional Assessment criteria** :

1. **Autonomous Control:**  Zero manual intervention during match play.  
2. **Multi-machine Coordination:**  Effective 3-unit tactical synchronization.  
3. **Model Robustness:**  Stability across diverse adversarial scenarios, autonomous stand-up within the competition limits, and the 10-minute shootout burst.**Engineering Lead Note:**  Adherence to these standards is mandatory to secure the Booster K1 Geek Edition grand prize. Focus on robustness; the simulation rewards stability over reckless speed.\# Technical Development & Participation Plan: MotrixArena S2 (Scheme A Implementation)

#### 1\. Executive Strategic Context

The MotrixArena S2 challenge represents the premier "Experimental Field for Embodied AI," providing a rigorous environment to transition abstract Reinforcement Learning (RL) algorithms into a high-fidelity 3v3 simulated soccer arena. This competition is a comprehensive stress test for autonomous control systems, requiring precise orchestration of motor skills under adversarial pressure. For the Senior Robotics Architect, this is not merely a game but a validation of the bridge between low-level locomotion and high-level tactical coordination.The engineering challenge is codified into four critical pillars that our architecture must address:

* **Stable Walking:**  Ensuring robust bipedal locomotion on a dynamic pitch.  
* **Multi-body Coordination:**  Synchronizing three units to maintain tactical integrity.  
* **Real-time Decision Making:**  Processing environment states to issue low-latency commands.  
* **Adversarial Games:**  Adapting to unpredictable opponent behaviors in real-time.Scheme A (K1 AMP) serves as our foundational solution. By leveraging a Full Joint Gait framework, we provide the necessary physical stability to support complex tactical maneuvers. Technical success is predicated on rigorous adherence to the platform's architectural constraints; non-compliance with the simulation's loading logic constitutes an immediate failure point.

#### 2\. Core Technical Architecture: Scheme A (K1 AMP) Specifications

Adopting Scheme A (K1 AMP / Full Joint Gait) is a strategic requirement to align with the native MotrixSim loading logic and the high-performance capabilities of the  **Booster K1 Geek Edition**  hardware. This alignment ensures that the gait model can fully utilize the physics engine's PD control gains for the 3v3 humanoid matches.

##### Technical Specification Table: Scheme A (K1 AMP)

Feature,Specification,Engineering Mandate  
Interface Shape,"N, 375 → N, 22",Inference Batching:  Typically export  N=1 . The simulation handles batching across robots internally.  
Observation Logic,5-frame stack (75 dims × 5),Mandatory 375-dimension input for state history.  
Observation Semantics,"Command, Gravity, Gyro, Joint Position Error, Joint Velocity, Previous Actions",Must follow K1\_amp\_config definitions in the training repo.  
Action Mapping,22-dimensional output,Must align precisely with K1\_AMP\_ACTUATOR\_JOINT\_ORDER and MJCF motor sequences.  
Control Format,.onnx (Recommended) or .pt,TorchScript is a valid fallback; ONNX is preferred for inference speed.  
Gait Switch,--k1-legged-gym,Must be enabled to prevent mixing with full-body AMP dimensions.

##### Architectural Insight: Scaling and Stability

The engineering team shall enforce the K1\_AMP\_ACTION\_SCALE \= 0.25 constant. This is not merely a limiter; it is a critical alignment factor for the physics engine's PD controller. A scale of 0.25 ensures that the output motor sequences stay within the MJCF motor limits, preventing actuator saturation and maintaining PD control stability during high-velocity transitions. Failure to apply this scale correctly will result in "posture collapse" or erratic oscillating behaviors. These motor skills are the execution arm for the Decider layer's movement vectors.

#### 3\. Decider-Gait Integration & Communication Protocol

Our architecture maintains a strict "Separation of Concerns." The Gait model handles the "How" (locomotion stability), while the Decider handles the "Where" (tactical positioning and ball interaction).

##### ZMQ Communication Layer

The Decider processes environmental states and issues velocity instructions (cmd) via ZMQ. The following JSON structure is mandatory for all team agents:  
{  
  "id": 0, // Red Team: 0-6; Blue Team: 7-13  
  "cmd": \[0.5, 0.1, 0.05\] // \[vx, vy, w\] normalized to robot coordinate system  
}

##### Protocol Compliance & Robustness

Engineering leads must account for the following communication risks:

1. **ID Allocation:**  Ensure Red Team agents use the 0-6 range and Blue Team use 7-13 to prevent cross-team command leakage.  
2. **Normalization & Dead Zones:**  Commands are subject to CMD\_VEL\_NORM clipping. The Decider must account for "Dead Zones" where low-magnitude commands (too small) will not trigger gait movement, and high-magnitude commands will be clipped by simulation logic (Section 4.2).  
3. **Written Confirmation:**  Any modifications to the ZMQ packet structure or JSON schema require written confirmation between the Strategy and Integration teams to prevent simulation-side parsing errors.

#### 4\. Environment Standardization & Deployment Readiness

To eliminate "works on my machine" integration failures, all developers must synchronize with the official MotrixArena S2 environment.

##### Deployment Requirements

* **Conda Environment:**  All inference and simulation tasks shall run within the motrixsim0508 environment.  
* **Decider Dependency Locking:**  All libraries must be frozen in decider/requirements.txt using pip freeze. Python version locking is non-negotiable.  
* **Inference Engine:**  onnxruntime is mandatory on the simulation side. When exporting models, the specific opset version used must be documented in the README\_Gait.md.  
* **Hardware Dependencies:**  If Decider logic utilizes GPU acceleration, the exact CUDA/cuDNN versions must be specified. Simulation ONNX inference defaults to CPU for maximum stability.  
* **Safe Paths Policy:**  All model and code paths must strictly forbid non-standard characters (Chinese characters, spaces). Non-compliance will cause path-resolution failures on the simulation server.

#### 5\. Quality Assurance & Self-Verification Protocol

The "Simulation Acceptance Criteria" are the final hurdle before submission. Every model must pass the following verification pipeline.

##### Pre-Submission Checklist

1. **Dimension Verification:**  Use onnxruntime to confirm single-input/single-output shapes of N, 375 and N, 22\.  
2. **Runtime Integrity:**  The Decider must demonstrate continuous state reception and cmd issuance without memory leaks or stack overflows over a 20-minute test loop.  
3. **Physical Verification:**  Use the \--policy flag to point to the model's absolute path and confirm the robot can maintain posture/walk using the \--k1-legged-gym switch.  
4. **Autonomous Fall-Recovery Test:**  Force front, back, and side falls and confirm that the submitted policy begins a stand-up attempt within 20 seconds, detects success or failure, limits failure handling to 3 consecutive attempts, and resumes locomotion after standing. Simulator-side teleportation or unconditional pose reset must be disabled for this test.
5. **Penalty and Return Test:**  Confirm that an Incapable Robot is removed for an initial 30-second penalty, that repeated violations accumulate penalty time, and that the released robot autonomously returns to the field.
6. **The Clean Machine Test:**  Reproduce the entire setup on a secondary, isolated machine using only the provided documentation and pip install \-r requirements.txt. If the setup is not reproducible in one step, it is rejected.

#### 6\. Competition Roadmap & Milestone Management

Strategic success requires temporal alignment with the MotrixArena S2 official broadcast schedule. We are preparing for a high-volume competition featuring 48 total matches in the Round Robin alone.

##### Competition Schedule (2026)

Stage,Date Range,Core Tasks & Volume  
Registration,May 1 – May 15,Team registration and intent filing.  
Model Submission,May 15 – June 5,"Finalization of RL models, .onnx exports, and README\_Gait.md."  
Qualifying (32-Team),June 8 – June 12,Initial performance screening and seed selection.  
Round Robin,June 12 – June 24,12 days of live broadcasts; 4 matches per day (48 total).  
Elimination Bracket,June 27 – July 3,"16-team through Semifinals; weekend ""Golden Time"" broadcasts."  
Grand Finals,July 5,"Finals, All-Star performance, and award ceremony."

##### Tactical Response: The Penalty Shootout

In the event of a draw, matches enter a "10-minute high-frequency burst" point-blank shootout. This requires the Gait controller to possess "Accumulated Error Reset" logic to prevent physical collapse during repetitive, high-intensity kicks. Our robustness strategy must prioritize posture recovery over aggressive movement during this phase.

##### Final Delivery & 3-Dimensional Assessment

All submissions must include a clear README\_Gait.md and exclude private tokens, logs, or /opt dependencies. Our final performance will be evaluated against the  **3-Dimensional Assessment criteria** :

1. **Autonomous Control:**  Zero manual intervention during match play.  
2. **Multi-machine Coordination:**  Effective 3-unit tactical synchronization.  
3. **Model Robustness:**  Stability across diverse adversarial scenarios, autonomous stand-up within the competition limits, and the 10-minute shootout burst.**Engineering Lead Note:**  Adherence to these standards is mandatory to secure the Booster K1 Geek Edition grand prize. Focus on robustness; the simulation rewards stability over reckless speed.
