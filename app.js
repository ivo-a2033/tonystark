import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const threeCanvas = document.querySelector("#three-overlay");
const context = canvas.getContext("2d");
const status = document.querySelector("#status");
const toggleButton = document.querySelector("#toggle-camera");
const modelInput = document.querySelector("#model-file");
const emptyState = document.querySelector("#empty-state");
const renderer = new THREE.WebGLRenderer({ canvas: threeCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(0, 1, 0, 1, 1, 2000);
camera.position.z = 1000;
const modelGroup = new THREE.Group();
const orientationGroup = new THREE.Group();
modelGroup.visible = false;
modelGroup.add(orientationGroup);
scene.add(modelGroup);

let handLandmarker;
let handModel;
let cameraStream;
let animationFrame;
let lastVideoTime = -1;
let lastPoseUpdateTime;
let poseInitialized = false;
let hasRightHandPose = false;
let targetScale = 1;
const targetHandAnchor = new THREE.Vector2();
const targetPosition = new THREE.Vector3();
const targetOrientation = new THREE.Quaternion();

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

async function initializeTracker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "./hand_landmarker.task" },
    runningMode: "VIDEO",
    numHands: 2,
  });

  status.textContent = "Choose an OBJ or DAE model";
  toggleButton.disabled = false;
}

function installModel(loadedHandModel) {
  if (handModel) {
    orientationGroup.remove(handModel);
    handModel.traverse((child) => {
      child.geometry?.dispose();
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) material?.dispose();
    });
  }

  handModel = loadedHandModel;
  poseInitialized = false;
  hasRightHandPose = false;
  targetScale = 1;
  handModel.rotation.x = -Math.PI / 2;
  handModel.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(handModel);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const baseSize = 150;
  handModel.position.sub(center);
  handModel.scale.setScalar(baseSize / Math.max(size.x, size.y, size.z));
  const meshes = [];
  handModel.traverse((child) => {
    if (child.isMesh) meshes.push(child);
  });

  const lineGroup = new THREE.Group();
  const lineMaterial = new LineMaterial({ color: 0x8f00f4, linewidth: 0.5 });
  handModel.updateMatrixWorld(true);
  const rootInverse = handModel.matrixWorld.clone().invert();
  for (const mesh of meshes) {
    const wireframe = new THREE.WireframeGeometry(mesh.geometry);
    const lineGeometry = new LineSegmentsGeometry().setPositions(
      Array.from(wireframe.attributes.position.array),
    );
    const lines = new LineSegments2(lineGeometry, lineMaterial);
    lines.matrixAutoUpdate = false;
    lines.matrix.copy(rootInverse.clone().multiply(mesh.matrixWorld));
    lineGroup.add(lines);
    mesh.visible = false;
    wireframe.dispose();
  }
  handModel.add(lineGroup);
  orientationGroup.add(handModel);
  modelGroup.visible = false;
}

async function loadSelectedModel(file) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  const contents = await file.text();
  let loadedHandModel;

  if (extension === "obj") {
    loadedHandModel = new OBJLoader().parse(contents);
  } else if (extension === "dae") {
    loadedHandModel = new ColladaLoader().parse(contents, "").scene;
  } else {
    throw new Error("Choose an OBJ or DAE file");
  }

  installModel(loadedHandModel);
  status.textContent = `Loaded ${file.name}`;
}

function resizeOverlay(width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  renderer.setSize(width, height, false);
  camera.left = 0;
  camera.right = width;
  camera.top = 0;
  camera.bottom = height;
  camera.updateProjectionMatrix();
}

function drawLandmarks(hands) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#35df74";
  context.lineWidth = Math.max(2, canvas.width / 640);
  context.fillStyle = "#35df74";

  for (const hand of hands) {
    context.beginPath();
    for (const [start, end] of HAND_CONNECTIONS) {
      context.moveTo(hand[start].x * canvas.width, hand[start].y * canvas.height);
      context.lineTo(hand[end].x * canvas.width, hand[end].y * canvas.height);
    }
    context.stroke();

    for (const landmark of hand) {
      context.beginPath();
      context.arc(landmark.x * canvas.width, landmark.y * canvas.height, Math.max(3, canvas.width / 320), 0, Math.PI * 2);
      context.fill();
    }
  }
}

function handForSide(hands, handedness, side) {
  // MediaPipe labels assume mirrored input; the camera frames here are unmirrored.
  const expectedLabel = side === "right" ? "Left" : "Right";
  const index = handedness.findIndex((categories) => categories[0]?.categoryName === expectedLabel);
  return hands[index];
}

function updateObject(hands, handedness, timestamp) {
  if (!handModel) {
    modelGroup.visible = false;
    return;
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  const rightHand = handForSide(hands, handedness, "right");
  const leftHand = handForSide(hands, handedness, "left");
  if (leftHand) {
    const thumbTip = leftHand[4];
    const indexTip = leftHand[8];
    const pinchDistance = Math.hypot(
      (thumbTip.x - indexTip.x) * width,
      (thumbTip.y - indexTip.y) * height,
    );
    targetScale = THREE.MathUtils.clamp(pinchDistance / (Math.min(width, height) * 0.16), 0.35, 2.5);
  }

  if (rightHand) {
    const toFramePoint = (landmark) => new THREE.Vector3(
      landmark.x * width,
      landmark.y * height,
      landmark.z * width,
    );
    const wrist = toFramePoint(rightHand[0]);
    const indexMcp = toFramePoint(rightHand[5]);
    const middleMcp = toFramePoint(rightHand[9]);
    const pinkyMcp = toFramePoint(rightHand[17]);
    const acrossAxis = pinkyMcp.sub(indexMcp);
    const longitudinalAxis = middleMcp.sub(wrist);
    if (acrossAxis.lengthSq() > 1e-6) {
      acrossAxis.normalize();
      longitudinalAxis.addScaledVector(acrossAxis, -longitudinalAxis.dot(acrossAxis));
      if (longitudinalAxis.lengthSq() > 1e-6) {
        longitudinalAxis.normalize();
        const normalAxis = acrossAxis.clone().cross(longitudinalAxis).normalize();
        const handFrame = new THREE.Matrix4().makeBasis(acrossAxis, longitudinalAxis, normalAxis);
        targetOrientation.setFromRotationMatrix(handFrame);
      }
    }

    const minY = Math.min(...rightHand.map((landmark) => landmark.y * height));
    targetHandAnchor.set(
      rightHand.reduce((sum, landmark) => sum + landmark.x, 0) / rightHand.length * width,
      minY,
    );
    hasRightHandPose = true;
  }

  if (!hasRightHandPose) {
    modelGroup.visible = false;
    return;
  }

  const objectHeight = 150 * targetScale;
  targetPosition.set(targetHandAnchor.x, targetHandAnchor.y - 18 - objectHeight / 2, 0);
  if (!poseInitialized) {
    modelGroup.position.copy(targetPosition);
    modelGroup.scale.setScalar(targetScale);
    orientationGroup.quaternion.copy(targetOrientation);
    poseInitialized = true;
  } else {
    const delta = lastPoseUpdateTime === undefined ? 0 : (timestamp - lastPoseUpdateTime) / 1000;
    const smoothing = 1 - Math.exp(-12 * Math.min(delta, 0.1));
    modelGroup.position.lerp(targetPosition, smoothing);
    modelGroup.scale.setScalar(THREE.MathUtils.lerp(modelGroup.scale.x, targetScale, smoothing));
    orientationGroup.quaternion.slerp(targetOrientation, smoothing);
  }
  lastPoseUpdateTime = timestamp;
  modelGroup.visible = true;
}

function detectFrame() {
  if (!cameraStream) return;

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    resizeOverlay(video.videoWidth, video.videoHeight);
    const result = handLandmarker.detectForVideo(video, performance.now());
    drawLandmarks(result.landmarks);
    updateObject(result.landmarks, result.handedness, performance.now());
    renderer.render(scene, camera);
    status.textContent = `${result.landmarks.length} hand${result.landmarks.length === 1 ? "" : "s"} detected`;
  }

  animationFrame = requestAnimationFrame(detectFrame);
}

async function startCamera() {
  toggleButton.disabled = true;
  status.textContent = "Requesting camera…";

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user" },
    });
    video.srcObject = cameraStream;
    await video.play();
    emptyState.hidden = true;
    toggleButton.textContent = "Stop camera";
    detectFrame();
  } catch (error) {
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
    status.textContent = error.name === "NotAllowedError" ? "Camera permission denied" : "Unable to start camera";
    console.error(error);
  } finally {
    toggleButton.disabled = false;
  }
}

function stopCamera() {
  cancelAnimationFrame(animationFrame);
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  emptyState.hidden = false;
  context.clearRect(0, 0, canvas.width, canvas.height);
  modelGroup.visible = false;
  renderer.clear();
  toggleButton.textContent = "Start camera";
  status.textContent = "Camera off";
}

toggleButton.addEventListener("click", () => {
  if (cameraStream) stopCamera();
  else startCamera();
});

modelInput.addEventListener("change", async () => {
  const file = modelInput.files?.[0];
  if (!file) return;

  try {
    await loadSelectedModel(file);
    if (!cameraStream && !toggleButton.disabled) await startCamera();
  } catch (error) {
    status.textContent = "Could not load model";
    console.error(error);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && cameraStream) stopCamera();
});

initializeTracker().catch((error) => {
  status.textContent = "Could not load hand tracking";
  console.error(error);
});