import "./index.css";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { motion } from "framer-motion";
  import { StrictMode, useMemo, useState } from "react";
  import { createRoot } from "react-dom/client";
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ReferenceDot,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";

/**
 * Free-return trajectory browser app
 *
 * Features:
 * - 3D two-body + third-body propagation in ECI coordinates
 * - Low-precision analytical lunar ephemeris (approximate real ephemerides, not circular)
 * - Earth rotation and launch-site geometry
 * - Single-parameter shooting method targeting chosen lunar perilune altitude
 * - Interactive plots and tabular trajectory samples
 *
 * Important limitations:
 * - This is not mission-grade. It does NOT use JPL DE ephemerides, high-order gravity, SRP,
 *   atmospheric drag, finite burns, or patched-conic targeting constraints.
 * - The Moon model here is a low-precision analytical ephemeris intended for visualization and
 *   design exploration, but it is still much more realistic than a circular planar Moon orbit.
 */

type Vec3 = { x: number; y: number; z: number };
type State = { r: Vec3; v: Vec3 };

type Sample = {
  tSec: number;
  utc: string;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  rEarth: number;
  rMoon: number;
  moonX: number;
  moonY: number;
  moonZ: number;
};

type SolveResult = {
  converged: boolean;
  message: string;
  dvAppliedKmS: number;
  moonPeriluneAltKm: number;
  moonPeriluneTimeHours: number;
  returnPerigeeAltKm: number;
  returnTimeHours: number;
  c3Km2S2: number;
  trajectory: Sample[];
  xyPlot: { x: number; y: number; label?: string }[];
  xzPlot: { x: number; z: number; label?: string }[];
  radiusPlot: { hours: number; rEarth: number; rMoon: number }[];
  xyMarkers: {
    x: number;
    y: number;
    moonX: number;
    moonY: number;
    hours: number;
    tag: string;
  }[];
  sideView: { x: number; z: number }[];
  plotBounds: {
    xy: { x: [number, number]; y: [number, number] };
    xz: { x: [number, number]; z: [number, number] };
  };
  keyEvents: { name: string; hours: number; value: number }[];
};

type Inputs = {
  departureUtc: string;
  launchLatDeg: number;
  launchLonDeg: number;
  launchAzimuthDeg: number;
  parkingAltKm: number;
  parkingPlaneInclinationDeg: number;
  targetMoonAltKm: number;
  initialDvGuessKmS: number;
  simDays: number;
  stepSec: number;
};

const MU_EARTH = 398600.4418;
const MU_MOON = 4902.800066;
const R_EARTH = 6378.1363;
const R_MOON = 1737.4;
const OMEGA_EARTH = 7.292115e-5; // rad/s
const DEG2RAD = Math.PI / 180;

function v3(x = 0, y = 0, z = 0): Vec3 {
  return { x, y, z };
}
function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}
function scale(a: Vec3, s: number): Vec3 {
  return { x: a.x * s, y: a.y * s, z: a.z * s };
}
function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}
function norm(a: Vec3): number {
  return Math.hypot(a.x, a.y, a.z);
}
function unit(a: Vec3): Vec3 {
  const n = norm(a);
  return n > 0 ? scale(a, 1 / n) : v3(0, 0, 0);
}
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
function fmt(x: number, digits = 3): string {
  return Number.isFinite(x) ? x.toFixed(digits) : "—";
}
function symmetricBounds(
  values: number[],
  pad = 0.12,
  minHalfRange = 25,
): [number, number] {
  const maxAbs = Math.max(minHalfRange, ...values.map((v) => Math.abs(v)));
  const r = maxAbs * (1 + pad);
  return [-r, r];
}

function julianDate(date: Date): number {
  return date.getTime() / 86400000 + 2440587.5;
}
function gmstRad(date: Date): number {
  const jd = julianDate(date);
  const T = (jd - 2451545.0) / 36525.0;
  let gmstDeg =
    280.46061837 +
    360.98564736629 * (jd - 2451545.0) +
    0.000387933 * T * T -
    (T * T * T) / 38710000.0;
  gmstDeg = ((gmstDeg % 360) + 360) % 360;
  return gmstDeg * DEG2RAD;
}
function rotateX(v: Vec3, a: number): Vec3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return { x: v.x, y: c * v.y - s * v.z, z: s * v.y + c * v.z };
}

function keplerSolve(M: number, e: number): number {
  let E = e < 0.8 ? M : Math.PI;
  for (let i = 0; i < 15; i++) {
    const f = E - e * Math.sin(E) - M;
    const fp = 1 - e * Math.cos(E);
    E -= f / fp;
  }
  return E;
}

function moonPositionECI(date: Date): Vec3 {
  // Low-precision analytical lunar ephemeris based on mean elements with leading periodic terms.
  // Suitable for visualization and rough trajectory design; not a replacement for DE430/440.
  const jd = julianDate(date);
  const d = jd - 2451543.5;

  const N = (125.1228 - 0.0529538083 * d) * DEG2RAD;
  const i = 5.1454 * DEG2RAD;
  const w = (318.0634 + 0.1643573223 * d) * DEG2RAD;
  const a = 60.2666 * R_EARTH;
  const e = 0.0549;
  const M = ((115.3654 + 13.0649929509 * d) % 360) * DEG2RAD;

  const E = keplerSolve(M, e);
  const xv = a * (Math.cos(E) - e);
  const yv = a * (Math.sqrt(1 - e * e) * Math.sin(E));
  const v = Math.atan2(yv, xv);
  const r = Math.hypot(xv, yv);

  const xh =
    r *
    (Math.cos(N) * Math.cos(v + w) -
      Math.sin(N) * Math.sin(v + w) * Math.cos(i));
  const yh =
    r *
    (Math.sin(N) * Math.cos(v + w) +
      Math.cos(N) * Math.sin(v + w) * Math.cos(i));
  const zh = r * (Math.sin(v + w) * Math.sin(i));

  // Convert from mean ecliptic-ish frame to equatorial using obliquity.
  const eps = (23.4393 - 3.563e-7 * d) * DEG2RAD;
  return rotateX(v3(xh, yh, zh), -eps);
}

function buildLaunchBasis(date: Date, latDeg: number, lonDeg: number) {
  const lat = latDeg * DEG2RAD;
  const lon = lonDeg * DEG2RAD;
  const theta = gmstRad(date) + lon;

  const up = v3(
    Math.cos(lat) * Math.cos(theta),
    Math.cos(lat) * Math.sin(theta),
    Math.sin(lat),
  );
  const east = unit(v3(-Math.sin(theta), Math.cos(theta), 0));
  const north = unit(cross(up, east));
  return { up, east, north };
}

function initialParkingState(inputs: Inputs): {
  state: State;
  tangential: Vec3;
} {
  const date = new Date(inputs.departureUtc);
  const basis = buildLaunchBasis(
    date,
    inputs.launchLatDeg,
    inputs.launchLonDeg,
  );

  const rMag = R_EARTH + inputs.parkingAltKm;
  const r0 = scale(basis.up, rMag);

  const az = inputs.launchAzimuthDeg * DEG2RAD;
  let tangential = unit(
    add(scale(basis.north, Math.cos(az)), scale(basis.east, Math.sin(az))),
  );

  // Tilt the local tangential direction to approximate the requested parking inclination.
  const localNormal = unit(cross(r0, tangential));
  const targetInc = inputs.parkingPlaneInclinationDeg * DEG2RAD;
  const currentInc = Math.acos(clamp(localNormal.z, -1, 1));
  const delta = targetInc - currentInc;
  tangential = unit(
    add(
      scale(tangential, Math.cos(delta)),
      scale(basis.north, Math.sin(delta) * 0.2),
    ),
  );

  const vCirc = Math.sqrt(MU_EARTH / rMag);
  const earthRot = cross(v3(0, 0, OMEGA_EARTH), r0);
  const v0 = add(scale(tangential, vCirc), earthRot);

  return { state: { r: r0, v: v0 }, tangential: unit(tangential) };
}

function accel(r: Vec3, tSec: number, departure: Date): Vec3 {
  const now = new Date(departure.getTime() + tSec * 1000);
  const rMoon = moonPositionECI(now);
  const relMoon = sub(r, rMoon);
  const dE = norm(r);
  const dM = norm(relMoon);
  const aEarth = scale(r, -MU_EARTH / Math.pow(dE, 3));
  const aMoon = scale(relMoon, -MU_MOON / Math.pow(dM, 3));
  return add(aEarth, aMoon);
}

function rk4Step(s: State, tSec: number, h: number, departure: Date): State {
  const a1 = accel(s.r, tSec, departure);
  const k1r = s.v;
  const k1v = a1;

  const s2 = { r: add(s.r, scale(k1r, h / 2)), v: add(s.v, scale(k1v, h / 2)) };
  const a2 = accel(s2.r, tSec + h / 2, departure);
  const k2r = s2.v;
  const k2v = a2;

  const s3 = { r: add(s.r, scale(k2r, h / 2)), v: add(s.v, scale(k2v, h / 2)) };
  const a3 = accel(s3.r, tSec + h / 2, departure);
  const k3r = s3.v;
  const k3v = a3;

  const s4 = { r: add(s.r, scale(k3r, h)), v: add(s.v, scale(k3v, h)) };
  const a4 = accel(s4.r, tSec + h, departure);
  const k4r = s4.v;
  const k4v = a4;

  return {
    r: add(s.r, scale(add(add(k1r, scale(add(k2r, k3r), 2)), k4r), h / 6)),
    v: add(s.v, scale(add(add(k1v, scale(add(k2v, k3v), 2)), k4v), h / 6)),
  };
}

function propagate(inputs: Inputs, dvKmS: number) {
  const departure = new Date(inputs.departureUtc);
  const { state: baseState, tangential } = initialParkingState(inputs);
  let s: State = {
    r: baseState.r,
    v: add(baseState.v, scale(tangential, dvKmS)),
  };

  const energy = 0.5 * dot(s.v, s.v) - MU_EARTH / norm(s.r);
  const c3 = 2 * energy;

  const totalTime = inputs.simDays * 86400;
  const h = inputs.stepSec;
  const saveEvery = Math.max(1, Math.round(1800 / h));
  const radiusPlotEvery = Math.max(1, Math.round(900 / h));

  const samples: Sample[] = [];
  const radiusPlot: { hours: number; rEarth: number; rMoon: number }[] = [];

  let minMoonDistance = Number.POSITIVE_INFINITY;
  let minMoonTime = 0;
  let minEarthAfterMoon = Number.POSITIVE_INFINITY;
  let minEarthAfterMoonTime = 0;
  let seenMoonEncounter = false;

  for (let step = 0, t = 0; t <= totalTime; step++, t += h) {
    const now = new Date(departure.getTime() + t * 1000);
    const rMoon = moonPositionECI(now);
    const relMoon = sub(s.r, rMoon);
    const dEarth = norm(s.r);
    const dMoon = norm(relMoon);

    if (dMoon < minMoonDistance) {
      minMoonDistance = dMoon;
      minMoonTime = t;
    }
    if (dMoon < 120000) seenMoonEncounter = true;
    if (seenMoonEncounter && dEarth < minEarthAfterMoon) {
      minEarthAfterMoon = dEarth;
      minEarthAfterMoonTime = t;
    }

    if (step % saveEvery === 0) {
      samples.push({
        tSec: t,
        utc: now.toISOString(),
        x: s.r.x,
        y: s.r.y,
        z: s.r.z,
        vx: s.v.x,
        vy: s.v.y,
        vz: s.v.z,
        rEarth: dEarth,
        rMoon: dMoon,
        moonX: rMoon.x,
        moonY: rMoon.y,
        moonZ: rMoon.z,
      });
    }

    if (step % radiusPlotEvery === 0) {
      radiusPlot.push({
        hours: t / 3600,
        rEarth: dEarth - R_EARTH,
        rMoon: dMoon - R_MOON,
      });
    }

    if (dEarth < R_EARTH || dMoon < R_MOON) {
      break;
    }

    s = rk4Step(s, t, h, departure);
  }

  const moonAlt = minMoonDistance - R_MOON;
  const earthReturnAlt = minEarthAfterMoon - R_EARTH;

  return {
    c3,
    moonAlt,
    moonTimeHours: minMoonTime / 3600,
    earthReturnAlt,
    earthReturnTimeHours: minEarthAfterMoonTime / 3600,
    samples,
    radiusPlot,
  };
}

function shootForPerilune(inputs: Inputs): SolveResult {
  let dv0 = inputs.initialDvGuessKmS - 0.04;
  let dv1 = inputs.initialDvGuessKmS + 0.04;

  let f0 = propagate(inputs, dv0).moonAlt - inputs.targetMoonAltKm;
  let f1 = propagate(inputs, dv1).moonAlt - inputs.targetMoonAltKm;

  let bestDv = inputs.initialDvGuessKmS;
  let bestErr = Number.POSITIVE_INFINITY;
  let bestRun = propagate(inputs, bestDv);

  for (const trial of [dv0, dv1, bestDv]) {
    const run = propagate(inputs, trial);
    const err = Math.abs(run.moonAlt - inputs.targetMoonAltKm);
    if (err < bestErr) {
      bestErr = err;
      bestDv = trial;
      bestRun = run;
    }
  }

  let converged = false;
  let message = "Reached iteration limit.";

  for (let iter = 0; iter < 12; iter++) {
    const denom = f1 - f0;
    let dv2 =
      Math.abs(denom) < 1e-6
        ? (dv0 + dv1) / 2
        : dv1 - (f1 * (dv1 - dv0)) / denom;
    dv2 = clamp(
      dv2,
      inputs.initialDvGuessKmS - 0.25,
      inputs.initialDvGuessKmS + 0.25,
    );

    const run2 = propagate(inputs, dv2);
    const f2 = run2.moonAlt - inputs.targetMoonAltKm;
    const err = Math.abs(f2);

    if (err < bestErr) {
      bestErr = err;
      bestDv = dv2;
      bestRun = run2;
    }

    if (err < 20) {
      converged = true;
      message = `Shooting converged in ${iter + 1} iterations.`;
      break;
    }

    dv0 = dv1;
    f0 = f1;
    dv1 = dv2;
    f1 = f2;
  }

  const xyPlot = bestRun.samples.map((s) => ({ x: s.x / 1000, y: s.y / 1000 }));
  const xzPlot = bestRun.samples.map((s) => ({ x: s.x / 1000, z: s.z / 1000 }));

  const markerCount = 8;
  const markerIdxs = Array.from({ length: markerCount }, (_, i) =>
    Math.min(
      bestRun.samples.length - 1,
      Math.round((i * (bestRun.samples.length - 1)) / (markerCount - 1)),
    ),
  );
  const xyMarkers = markerIdxs.map((idx, i) => {
    const s = bestRun.samples[idx];
    return {
      x: s.x / 1000,
      y: s.y / 1000,
      moonX: s.moonX / 1000,
      moonY: s.moonY / 1000,
      hours: s.tSec / 3600,
      tag: `T${i}`,
    };
  });

  const sideView = bestRun.samples.map((s) => ({
    x: s.x / 1000,
    z: s.z / 1000,
  }));

  const xyX = [
    ...xyPlot.map((p) => p.x),
    ...xyMarkers.map((m) => m.moonX),
    ...xyMarkers.map((m) => m.x),
    0,
  ];
  const xyY = [
    ...xyPlot.map((p) => p.y),
    ...xyMarkers.map((m) => m.moonY),
    ...xyMarkers.map((m) => m.y),
    0,
  ];
  const xzX = [...sideView.map((p) => p.x), 0];
  const xzZ = [...sideView.map((p) => p.z), 0];

  return {
    converged,
    message,
    dvAppliedKmS: bestDv,
    moonPeriluneAltKm: bestRun.moonAlt,
    moonPeriluneTimeHours: bestRun.moonTimeHours,
    returnPerigeeAltKm: bestRun.earthReturnAlt,
    returnTimeHours: bestRun.earthReturnTimeHours,
    c3Km2S2: bestRun.c3,
    trajectory: bestRun.samples,
    xyPlot,
    xzPlot,
    radiusPlot: bestRun.radiusPlot,
    xyMarkers,
    sideView,
    plotBounds: {
      xy: { x: symmetricBounds(xyX), y: symmetricBounds(xyY) },
      xz: { x: symmetricBounds(xzX), z: symmetricBounds(xzZ) },
    },
    keyEvents: [
      {
        name: "Moon perilune",
        hours: bestRun.moonTimeHours,
        value: bestRun.moonAlt,
      },
      {
        name: "Earth return",
        hours: bestRun.earthReturnTimeHours,
        value: bestRun.earthReturnAlt,
      },
    ],
  };
}

const defaultInputs: Inputs = {
  departureUtc: "2026-05-10T12:00:00.000Z",
  launchLatDeg: 28.5729,
  launchLonDeg: -80.649,
  launchAzimuthDeg: 90,
  parkingAltKm: 185,
  parkingPlaneInclinationDeg: 28.5,
  targetMoonAltKm: 3000,
  initialDvGuessKmS: 3.15,
  simDays: 9,
  stepSec: 30,
};

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border p-3 shadow-sm">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}

export default function FreeReturnTrajectoryWebapp() {
  const [inputs, setInputs] = useState<Inputs>(defaultInputs);
  const [result, setResult] = useState<SolveResult | null>(() =>
    shootForPerilune(defaultInputs),
  );

  const moonAtDeparture = useMemo(
    () => moonPositionECI(new Date(inputs.departureUtc)),
    [inputs.departureUtc],
  );

  function patch<K extends keyof Inputs>(key: K, value: Inputs[K]) {
    setInputs((s) => ({ ...s, [key]: value }));
  }

  function runSolve() {
    setResult(shootForPerilune(inputs));
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto grid max-w-7xl gap-6">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="rounded-3xl shadow-lg">
            <CardHeader>
              <CardTitle className="text-3xl">
                Free-return trajectory explorer
              </CardTitle>
              <div className="max-w-4xl text-sm text-muted-foreground">
                3D Earth–Moon propagation with Earth rotation, launch-site
                geometry, low-precision analytical lunar ephemerides, and a
                secant shooting method to target a chosen lunar perilune
                altitude. The plots and table below are generated from the
                solved trajectory.
              </div>
            </CardHeader>
          </Card>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-[380px,1fr]">
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <Card className="rounded-3xl shadow-lg">
              <CardHeader>
                <CardTitle>Inputs</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="grid gap-2">
                  <Label>Departure UTC</Label>
                  <Input
                    value={inputs.departureUtc}
                    onChange={(e) => patch("departureUtc", e.target.value)}
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <Label>Launch latitude</Label>
                    <Input
                      type="number"
                      value={inputs.launchLatDeg}
                      onChange={(e) =>
                        patch("launchLatDeg", Number(e.target.value))
                      }
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label>Launch longitude</Label>
                    <Input
                      type="number"
                      value={inputs.launchLonDeg}
                      onChange={(e) =>
                        patch("launchLonDeg", Number(e.target.value))
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <Label>Launch azimuth</Label>
                    <Input
                      type="number"
                      value={inputs.launchAzimuthDeg}
                      onChange={(e) =>
                        patch("launchAzimuthDeg", Number(e.target.value))
                      }
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label>Parking inclination</Label>
                    <Input
                      type="number"
                      value={inputs.parkingPlaneInclinationDeg}
                      onChange={(e) =>
                        patch(
                          "parkingPlaneInclinationDeg",
                          Number(e.target.value),
                        )
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <Label>Parking altitude (km)</Label>
                    <Input
                      type="number"
                      value={inputs.parkingAltKm}
                      onChange={(e) =>
                        patch("parkingAltKm", Number(e.target.value))
                      }
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label>Target Moon alt (km)</Label>
                    <Input
                      type="number"
                      value={inputs.targetMoonAltKm}
                      onChange={(e) =>
                        patch("targetMoonAltKm", Number(e.target.value))
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <Label>Initial dV guess (km/s)</Label>
                    <Input
                      type="number"
                      step="0.01"
                      value={inputs.initialDvGuessKmS}
                      onChange={(e) =>
                        patch("initialDvGuessKmS", Number(e.target.value))
                      }
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label>Simulation days</Label>
                    <Input
                      type="number"
                      value={inputs.simDays}
                      onChange={(e) => patch("simDays", Number(e.target.value))}
                    />
                  </div>
                </div>
                <div className="grid gap-2">
                  <Label>Integrator step (sec)</Label>
                  <Input
                    type="number"
                    value={inputs.stepSec}
                    onChange={(e) => patch("stepSec", Number(e.target.value))}
                  />
                </div>
                <Button className="rounded-2xl" onClick={runSolve}>
                  Solve and plot
                </Button>
                <div className="rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                  Moon at departure: x={fmt(moonAtDeparture.x, 0)} km, y=
                  {fmt(moonAtDeparture.y, 0)} km, z={fmt(moonAtDeparture.z, 0)}{" "}
                  km.
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <div className="grid gap-6">
            <motion.div
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <Card className="rounded-3xl shadow-lg">
                <CardHeader>
                  <CardTitle>Solution summary</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-4">
                  {result && (
                    <>
                      <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
                        <SmallStat
                          label="Status"
                          value={result.converged ? "Converged" : "Best effort"}
                        />
                        <SmallStat
                          label="Applied dV"
                          value={`${fmt(result.dvAppliedKmS, 4)} km/s`}
                        />
                        <SmallStat
                          label="Moon perilune alt"
                          value={`${fmt(result.moonPeriluneAltKm, 1)} km`}
                        />
                        <SmallStat
                          label="Moon perilune time"
                          value={`${fmt(result.moonPeriluneTimeHours, 2)} h`}
                        />
                        <SmallStat
                          label="Earth return alt"
                          value={`${fmt(result.returnPerigeeAltKm, 1)} km`}
                        />
                        <SmallStat
                          label="C3"
                          value={`${fmt(result.c3Km2S2, 3)} km²/s²`}
                        />
                      </div>
                      <div className="rounded-2xl border p-3 text-sm text-muted-foreground">
                        {result.message}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </motion.div>

            <div className="grid gap-6 xl:grid-cols-2">
              <Card className="rounded-3xl shadow-lg">
                <CardHeader>
                  <CardTitle>Top-down view: Earth-centered X–Y</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-3 text-sm text-muted-foreground">
                    Spacecraft path around Earth in the X–Y plane, with labeled
                    spacecraft and Moon positions at selected timesteps.
                  </div>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={result?.xyPlot ?? []}
                        margin={{ top: 16, right: 24, bottom: 24, left: 24 }}
                      >
                        <CartesianGrid />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={result?.plotBounds.xy.x ?? ["auto", "auto"]}
                          unit=" kkm"
                          tickFormatter={(v) => `${v}`}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          domain={result?.plotBounds.xy.y ?? ["auto", "auto"]}
                          unit=" kkm"
                          tickFormatter={(v) => `${v}`}
                        />
                        <Tooltip
                          formatter={(value: number) => fmt(Number(value), 2)}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="y"
                          name="Spacecraft trajectory"
                          strokeWidth={3}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <ReferenceDot x={0} y={0} r={7} label="Earth" />
                        {(result?.xyMarkers ?? []).map((m) => (
                          <ReferenceDot
                            key={`sc-${m.tag}`}
                            x={m.x}
                            y={m.y}
                            r={5}
                            label={{ value: m.tag, position: "top" }}
                          />
                        ))}
                        {(result?.xyMarkers ?? []).map((m) => (
                          <ReferenceDot
                            key={`moon-${m.tag}`}
                            x={m.moonX}
                            y={m.moonY}
                            r={4}
                            label={{ value: `${m.tag}M`, position: "bottom" }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="rounded-3xl shadow-lg">
                <CardHeader>
                  <CardTitle>Side view: Earth-centered X–Z</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-3 text-sm text-muted-foreground">
                    Simplified side profile of the spacecraft trajectory
                    relative to Earth only.
                  </div>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={result?.sideView ?? []}
                        margin={{ top: 16, right: 24, bottom: 24, left: 24 }}
                      >
                        <CartesianGrid />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={result?.plotBounds.xz.x ?? ["auto", "auto"]}
                          unit=" kkm"
                          tickFormatter={(v) => `${v}`}
                        />
                        <YAxis
                          type="number"
                          dataKey="z"
                          domain={result?.plotBounds.xz.z ?? ["auto", "auto"]}
                          unit=" kkm"
                          tickFormatter={(v) => `${v}`}
                        />
                        <Tooltip
                          formatter={(value: number) => fmt(Number(value), 2)}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="z"
                          name="Trajectory side view"
                          strokeWidth={3}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <ReferenceDot x={0} y={0} r={7} label="Earth" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="rounded-3xl shadow-lg">
              <CardHeader>
                <CardTitle>Marked timesteps</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-52 overflow-auto rounded-2xl border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Tag</TableHead>
                        <TableHead>Time (h)</TableHead>
                        <TableHead>SC X (kkm)</TableHead>
                        <TableHead>SC Y (kkm)</TableHead>
                        <TableHead>Moon X (kkm)</TableHead>
                        <TableHead>Moon Y (kkm)</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(result?.xyMarkers ?? []).map((m) => (
                        <TableRow key={m.tag}>
                          <TableCell>{m.tag}</TableCell>
                          <TableCell>{fmt(m.hours, 2)}</TableCell>
                          <TableCell>{fmt(m.x, 2)}</TableCell>
                          <TableCell>{fmt(m.y, 2)}</TableCell>
                          <TableCell>{fmt(m.moonX, 2)}</TableCell>
                          <TableCell>{fmt(m.moonY, 2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-3xl shadow-lg">
              <CardHeader>
                <CardTitle>Radius history</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={result?.radiusPlot ?? []}
                      margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
                    >
                      <CartesianGrid />
                      <XAxis dataKey="hours" unit=" h" />
                      <YAxis unit=" km" />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="rEarth"
                        dot={false}
                        name="Altitude above Earth"
                      />
                      <Line
                        type="monotone"
                        dataKey="rMoon"
                        dot={false}
                        name="Altitude above Moon"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-3xl shadow-lg">
              <CardHeader>
                <CardTitle>Trajectory samples</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-105 overflow-auto rounded-2xl border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>UTC</TableHead>
                        <TableHead>X (km)</TableHead>
                        <TableHead>Y (km)</TableHead>
                        <TableHead>Z (km)</TableHead>
                        <TableHead>Vx</TableHead>
                        <TableHead>Vy</TableHead>
                        <TableHead>Vz</TableHead>
                        <TableHead>rEarth</TableHead>
                        <TableHead>rMoon</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(result?.trajectory ?? [])
                        .slice(0, 200)
                        .map((row) => (
                          <TableRow key={row.tSec}>
                            <TableCell className="font-mono text-xs">
                              {row.utc}
                            </TableCell>
                            <TableCell>{fmt(row.x, 1)}</TableCell>
                            <TableCell>{fmt(row.y, 1)}</TableCell>
                            <TableCell>{fmt(row.z, 1)}</TableCell>
                            <TableCell>{fmt(row.vx, 4)}</TableCell>
                            <TableCell>{fmt(row.vy, 4)}</TableCell>
                            <TableCell>{fmt(row.vz, 4)}</TableCell>
                            <TableCell>{fmt(row.rEarth, 1)}</TableCell>
                            <TableCell>{fmt(row.rMoon, 1)}</TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  Showing up to the first 200 saved samples. The internal
                  propagator may evaluate many more steps depending on the
                  integrator step size.
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element #root was not found.");
}

createRoot(rootElement).render(
  <StrictMode>
    <FreeReturnTrajectoryWebapp />
  </StrictMode>,
);
