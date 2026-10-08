import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const threeCanvas = document.querySelector("#three-overlay");
const context = canvas.getContext("2d");
const status = document.querySelector("#status");
const toggleButton = document.querySelector("#toggle-camera");
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

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

async function loadModel() {
  const [vision, loadedHandModel] = await Promise.all([
    FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
    ),
    new OBJLoader().loadAsync("./copyrightFreeToaster.obj"),
  ]);

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "./hand_landmarker.task" },
    runningMode: "VIDEO",
    numHands: 2,
  });

  handModel = loadedHandModel;
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
  const lineMaterial = new LineMaterial({ color: 0x8ff0a4, linewidth: 0.5 });
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

  status.textContent = "Camera off";
  toggleButton.disabled = false;
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

function updateObject(hands, handedness) {
  if (!handModel) {
    modelGroup.visible = false;
    return;
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  const rightHand = handForSide(hands, handedness, "right");
  if (!rightHand) {
    modelGroup.visible = false;
    return;
  }

  const leftHand = handForSide(hands, handedness, "left");
  let scale = 1;
  if (leftHand) {
    const thumbTip = leftHand[4];
    const indexTip = leftHand[8];
    const pinchDistance = Math.hypot(
      (thumbTip.x - indexTip.x) * width,
      (thumbTip.y - indexTip.y) * height,
    );
    scale = THREE.MathUtils.clamp(pinchDistance / (Math.min(width, height) * 0.16), 0.35, 2.5);
  }

  const toFramePoint = (landmark) => new THREE.Vector3(
    landmark.x * width,
    landmark.y * height,
    landmark.z * width,
  );
  const wrist = toFramePoint(rightHand[0]);
  const indexMcp = rightHand[5];
  const middleMcp = rightHand[9];
  const pinkyMcp = rightHand[17];
  const acrossAxis = toFramePoint(pinkyMcp).sub(toFramePoint(indexMcp)).normalize();
  const longitudinalAxis = toFramePoint(middleMcp).sub(wrist);
  longitudinalAxis.addScaledVector(acrossAxis, -longitudinalAxis.dot(acrossAxis)).normalize();
  const normalAxis = acrossAxis.clone().cross(longitudinalAxis).normalize();
  if (acrossAxis.lengthSq() > 0.5 && longitudinalAxis.lengthSq() > 0.5 && normalAxis.lengthSq() > 0.5) {
    const handFrame = new THREE.Matrix4().makeBasis(acrossAxis, longitudinalAxis, normalAxis);
    orientationGroup.quaternion.setFromRotationMatrix(handFrame);
  }

  const minY = Math.min(...rightHand.map((landmark) => landmark.y * height));
  const centerX = rightHand.reduce((sum, landmark) => sum + landmark.x, 0) / rightHand.length * width;
  const objectHeight = 150 * scale;
  modelGroup.position.set(centerX, minY - 18 - objectHeight / 2, 0);
  modelGroup.scale.setScalar(scale);
  modelGroup.visible = true;
}

function detectFrame() {
  if (!cameraStream) return;

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    resizeOverlay(video.videoWidth, video.videoHeight);
    const result = handLandmarker.detectForVideo(video, performance.now());
    drawLandmarks(result.landmarks);
    updateObject(result.landmarks, result.handedness);
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

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && cameraStream) stopCamera();
});

loadModel().catch((error) => {
  status.textContent = "Could not load tracking models";
  console.error(error);
});