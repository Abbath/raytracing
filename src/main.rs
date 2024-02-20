use std::ops;

static EPSILON: f32 = 0.00001;

#[derive(Debug, Clone, Copy)]
struct Tuple {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

use Tuple as Point;
use Tuple as Vector;
use Tuple as Color;

impl Tuple {
    fn is_point(self) -> bool {
        self.w == 1.0
    }
    fn is_vector(self) -> bool {
        self.w == 0.0
    }
    fn magnitude(self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt()
    }
    fn normalize(self) -> Vector {
        let m = self.magnitude();
        Vector {
            x: self.x / m,
            y: self.y / m,
            z: self.z / m,
            w: self.w / m,
        }
    }
    fn dot(self, other: Vector) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
    fn cross(self, other: Vector) -> Vector {
        vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    fn reflect(self, normal: Vector) -> Vector {
        self - normal * 2.0 * self.dot(normal)
    }
}

impl PartialEq<Tuple> for Tuple {
    fn eq(&self, other: &Tuple) -> bool {
        let f = |a: f32, b: f32| (a - b).abs() < EPSILON;
        f(self.x, other.x) && f(self.y, other.y) && f(self.z, other.z) && self.w == other.w
    }
}

impl ops::Add<Tuple> for Tuple {
    type Output = Tuple;
    fn add(self, other: Tuple) -> Tuple {
        Tuple {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl ops::Sub<Tuple> for Tuple {
    type Output = Tuple;
    fn sub(self, other: Tuple) -> Tuple {
        Tuple {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl ops::Mul<f32> for Tuple {
    type Output = Tuple;
    fn mul(self, scale: f32) -> Tuple {
        Tuple {
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
            w: self.w * scale,
        }
    }
}

impl ops::Div<f32> for Tuple {
    type Output = Tuple;
    fn div(self, scale: f32) -> Tuple {
        Tuple {
            x: self.x / scale,
            y: self.y / scale,
            z: self.z / scale,
            w: self.w / scale,
        }
    }
}

impl ops::Neg for Tuple {
    type Output = Tuple;
    fn neg(self) -> Self::Output {
        Tuple {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl ops::Mul<Tuple> for Tuple {
    type Output = Tuple;
    fn mul(self, other: Tuple) -> Tuple {
        Tuple {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }
}

fn tuple(x: f32, y: f32, z: f32, w: f32) -> Tuple {
    Tuple { x, y, z, w }
}

fn point(x: f32, y: f32, z: f32) -> Point {
    Point { x, y, z, w: 1.0f32 }
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector { x, y, z, w: 0.0f32 }
}

fn color(x: f32, y: f32, z: f32) -> Color {
    Color { x, y, z, w: 0.0f32 }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Matrix4x4<T: Copy> {
    data: [T; 16],
}

#[derive(Debug, Clone, PartialEq)]
struct Matrix3x3<T: Copy> {
    data: [T; 9],
}

#[derive(Debug, Clone, PartialEq)]
struct Matrix2x2<T: Copy> {
    data: [T; 4],
}

fn identity_matrix() -> Matrix4x4<f32> {
    Matrix4x4 {
        data: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    }
}

fn translation(x: f32, y: f32, z: f32) -> Matrix4x4<f32> {
    let mut id = identity_matrix();
    id.set(0, 3, x);
    id.set(1, 3, y);
    id.set(2, 3, z);
    id
}

fn scaling(x: f32, y: f32, z: f32) -> Matrix4x4<f32> {
    let mut id = identity_matrix();
    id.set(0, 0, x);
    id.set(1, 1, y);
    id.set(2, 2, z);
    id
}

fn rotation_x(a: f32) -> Matrix4x4<f32> {
    let mut id = identity_matrix();
    id.set(1, 1, a.cos());
    id.set(1, 2, -a.sin());
    id.set(2, 1, a.sin());
    id.set(2, 2, a.cos());
    id
}

fn rotation_y(a: f32) -> Matrix4x4<f32> {
    let mut id = identity_matrix();
    id.set(0, 0, a.cos());
    id.set(0, 2, a.sin());
    id.set(2, 0, -a.sin());
    id.set(2, 2, a.cos());
    id
}

fn rotation_z(a: f32) -> Matrix4x4<f32> {
    let mut id = identity_matrix();
    id.set(0, 0, a.cos());
    id.set(0, 1, -a.sin());
    id.set(1, 0, a.sin());
    id.set(1, 1, a.cos());
    id
}

fn shearing(x_y: f32, x_z: f32, y_x: f32, y_z: f32, z_x: f32, z_y: f32) -> Matrix4x4<f32> {
    let mut id = identity_matrix();
    id.set(0, 1, x_y);
    id.set(0, 2, x_z);
    id.set(1, 0, y_x);
    id.set(1, 2, y_z);
    id.set(2, 0, z_x);
    id.set(2, 1, z_y);
    id
}

impl<T: Copy> Matrix2x2<T> {
    fn set(&mut self, r: usize, c: usize, val: T) {
        let idx = r * 2 + c;
        self.data[idx] = val;
    }
}

impl<T: Copy> Matrix3x3<T> {
    fn at(&self, r: usize, c: usize) -> T {
        let idx = r * 3 + c;
        self.data[idx]
    }
    fn set(&mut self, r: usize, c: usize, val: T) {
        let idx = r * 3 + c;
        self.data[idx] = val;
    }
    fn submatrix(&self, r: usize, c: usize) -> Matrix2x2<T> {
        let mut m = Matrix2x2 {
            data: [self.data[0]; 4],
        };
        let mut ro = 0;
        let mut co;
        for row in 0..3 {
            if row != r {
                co = 0;
                for col in 0..3 {
                    if col != c {
                        m.set(ro, co, self.at(row, col));
                        co += 1;
                    }
                }
                ro += 1;
            }
        }
        m
    }
}

impl<T: Copy> Matrix4x4<T> {
    fn at(&self, r: usize, c: usize) -> T {
        let idx = r * 4 + c;
        self.data[idx]
    }
    fn set(&mut self, r: usize, c: usize, val: T) {
        let idx = r * 4 + c;
        self.data[idx] = val;
    }
    fn transpose(&mut self) {
        let mut v = [self.data[0]; 16];
        v[0] = self.data[0];
        v[1] = self.data[4];
        v[2] = self.data[8];
        v[3] = self.data[12];
        v[4] = self.data[1];
        v[5] = self.data[5];
        v[6] = self.data[9];
        v[7] = self.data[13];
        v[8] = self.data[2];
        v[9] = self.data[6];
        v[10] = self.data[10];
        v[11] = self.data[14];
        v[12] = self.data[3];
        v[13] = self.data[7];
        v[14] = self.data[11];
        v[15] = self.data[15];
        self.data = v;
    }
    fn transposed(&self) -> Matrix4x4<T> {
        let mut v = [self.data[0]; 16];
        v[0] = self.data[0];
        v[1] = self.data[4];
        v[2] = self.data[8];
        v[3] = self.data[12];
        v[4] = self.data[1];
        v[5] = self.data[5];
        v[6] = self.data[9];
        v[7] = self.data[13];
        v[8] = self.data[2];
        v[9] = self.data[6];
        v[10] = self.data[10];
        v[11] = self.data[14];
        v[12] = self.data[3];
        v[13] = self.data[7];
        v[14] = self.data[11];
        v[15] = self.data[15];
        Matrix4x4 { data: v }
    }
    fn submatrix(&self, r: usize, c: usize) -> Matrix3x3<T> {
        let mut m = Matrix3x3 {
            data: [self.data[0]; 9],
        };
        let mut ro = 0;
        let mut co;
        for row in 0..4 {
            if row != r {
                co = 0;
                for col in 0..4 {
                    if col != c {
                        m.set(ro, co, self.at(row, col));
                        co += 1;
                    }
                }
                ro += 1;
            }
        }
        m
    }
}

impl Matrix3x3<f32> {
    fn determinant(&self) -> f32 {
        (0usize..3)
            .map(|c| self.at(0, c) * self.cofactor(0, c))
            .sum()
    }
    fn minor(&self, r: usize, c: usize) -> f32 {
        self.submatrix(r, c).determinant()
    }
    fn cofactor(&self, r: usize, c: usize) -> f32 {
        self.minor(r, c) * if (r + c) % 2 == 1 { -1.0 } else { 1.0 }
    }
}

impl Matrix2x2<f32> {
    fn determinant(&self) -> f32 {
        self.data[0] * self.data[3] - self.data[1] * self.data[2]
    }
}

impl Matrix4x4<f32> {
    fn determinant(&self) -> f32 {
        (0usize..4)
            .map(|c| self.at(0, c) * self.cofactor(0, c))
            .sum()
    }
    fn minor(&self, r: usize, c: usize) -> f32 {
        self.submatrix(r, c).determinant()
    }
    fn cofactor(&self, r: usize, c: usize) -> f32 {
        self.minor(r, c) * if (r + c) % 2 == 1 { -1.0 } else { 1.0 }
    }
    fn inverse(&self) -> Matrix4x4<f32> {
        let d = self.determinant();
        if d == 0.0 {
            panic!("Not invertible");
        }
        let mut m = self.clone();
        for row in 0..4 {
            for col in 0..4 {
                let c = self.cofactor(row, col);
                m.set(col, row, c / d);
            }
        }
        m
    }
}

impl ops::Mul<Matrix4x4<f32>> for Matrix4x4<f32> {
    type Output = Matrix4x4<f32>;
    fn mul(self, other: Matrix4x4<f32>) -> Matrix4x4<f32> {
        let mut m = Matrix4x4::<f32> { data: [0f32; 16] };
        for row in 0usize..=3usize {
            for col in 0usize..=3usize {
                m.set(
                    row,
                    col,
                    self.at(row, 0) * other.at(0, col)
                        + self.at(row, 1) * other.at(1, col)
                        + self.at(row, 2) * other.at(2, col)
                        + self.at(row, 3) * other.at(3, col),
                );
            }
        }
        m
    }
}

impl ops::Mul<Tuple> for Matrix4x4<f32> {
    type Output = Tuple;
    fn mul(self, other: Tuple) -> Tuple {
        let mut t = tuple(0.0, 0.0, 0.0, 0.0);
        t.x = self.at(0, 0) * other.x
            + self.at(0, 1) * other.y
            + self.at(0, 2) * other.z
            + self.at(0, 3) * other.w;
        t.y = self.at(1, 0) * other.x
            + self.at(1, 1) * other.y
            + self.at(1, 2) * other.z
            + self.at(1, 3) * other.w;
        t.z = self.at(2, 0) * other.x
            + self.at(2, 1) * other.y
            + self.at(2, 2) * other.z
            + self.at(2, 3) * other.w;
        t.w = self.at(3, 0) * other.x
            + self.at(3, 1) * other.y
            + self.at(3, 2) * other.z
            + self.at(3, 3) * other.w;
        t
    }
}

struct Canvas {
    m: Vec<Tuple>,
    w: usize,
    h: usize,
}

fn clamp_color(c: f32) -> u8 {
    if c < 0.0 {
        0
    } else if c > 255.0 {
        255
    } else {
        c as u8
    }
}

impl Canvas {
    fn pixel_at(self, x: usize, y: usize) -> Color {
        let idx = y * self.w + x;
        self.m[idx]
    }
    fn write_pixel(&mut self, x: usize, y: usize, c: Color) {
        let idx = y * self.w + x;
        self.m[idx] = c;
    }
    fn to_ppm(&self) -> String {
        let mut s = "P3\n".to_string();
        s += &format!("{} {}\n255\n", self.w, self.h);
        for c in self.m.iter() {
            s += &format!(
                "{} {} {}\n",
                clamp_color(c.x * 256.0),
                clamp_color(c.y * 256.0),
                clamp_color(c.z * 256.0)
            );
        }
        s
    }
}

fn canvas(w: usize, h: usize) -> Canvas {
    Canvas {
        m: vec![color(0.0, 0.0, 0.0); w * h],
        w,
        h,
    }
}

#[derive(Debug, Clone, Copy)]
struct Ray {
    origin: Point,
    direction: Vector,
}

fn ray(origin: Point, direction: Vector) -> Ray {
    Ray { origin, direction }
}

impl Ray {
    fn position(&self, t: f32) -> Point {
        self.origin + self.direction * t
    }
    fn transform(&self, t: Matrix4x4<f32>) -> Ray {
        let o = t * self.origin;
        let d = t * self.direction;
        Ray {
            origin: o,
            direction: d,
        }
    }
    fn check_cap_cylinder(&self, t: f32) -> bool {
        let x = self.origin.x + t * self.direction.x;
        let z = self.origin.z + t * self.direction.z;
        (x.powi(2) + z.powi(2)) - 1.0 < EPSILON
    }
    fn check_cap_cone(&self, t: f32) -> bool {
        let x = self.origin.x + t * self.direction.x;
        let z = self.origin.z + t * self.direction.z;
        let y = self.origin.y + t * self.direction.y;
        (x.powi(2) + z.powi(2)) - y.abs() < EPSILON
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ShapeType {
    Sphere,
    Plane,
    Cube,
    Cylinder(f32, f32, bool),
    Cone(f32, f32, bool),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Shape {
    typ: ShapeType,
    transform: Matrix4x4<f32>,
    material: Material,
}

fn sphere() -> Shape {
    Shape {
        typ: ShapeType::Sphere,
        transform: identity_matrix(),
        material: material(),
    }
}

fn glass_sphere() -> Shape {
    let mut s = sphere();
    s.material.transparency = 1.0;
    s.material.refractive_index = 1.5;
    s
}

fn check_axis(origin: f32, direction: f32) -> (f32, f32) {
    let tmin_numerator = -1.0 - origin;
    let tmax_numerator = 1.0 - origin;

    let (mut tmin, mut tmax) = if direction.abs() >= EPSILON {
        (tmin_numerator / direction, tmax_numerator / direction)
    } else {
        (
            tmin_numerator * f32::INFINITY,
            tmax_numerator * f32::INFINITY,
        )
    };

    if tmin > tmax {
        (tmin, tmax) = (tmax, tmin);
    }

    (tmin, tmax)
}

impl Shape {
    fn get_material(&self) -> Material {
        self.material
    }
    fn set_material(&mut self, m: Material) {
        self.material = m;
    }
    fn get_transform(&self) -> Matrix4x4<f32> {
        self.transform
    }
    fn set_transform(&mut self, t: Matrix4x4<f32>) {
        self.transform = t;
    }
    fn local_intersect(&self, r: Ray) -> Vec<Intersection> {
        match self.typ {
            ShapeType::Sphere => {
                let sphere_to_ray = r.origin - point(0.0, 0.0, 0.0);
                let a = r.direction.dot(r.direction);
                let b = 2.0 * r.direction.dot(sphere_to_ray);
                let c = sphere_to_ray.dot(sphere_to_ray) - 1.0;
                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    return intersections![];
                }
                let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
                let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
                intersections![intersection(t1, *self), intersection(t2, *self)]
            }
            ShapeType::Plane => {
                if r.direction.y.abs() < EPSILON {
                    vec![]
                } else {
                    let t = -r.origin.y / r.direction.y;
                    vec![intersection(t, *self)]
                }
            }
            ShapeType::Cube => {
                let (xtmin, xtmax) = check_axis(r.origin.x, r.direction.x);
                let (ytmin, ytmax) = check_axis(r.origin.y, r.direction.y);
                let (ztmin, ztmax) = check_axis(r.origin.z, r.direction.z);

                let tmin = xtmin.max(ytmin.max(ztmin));
                let tmax = xtmax.min(ytmax.min(ztmax));

                if tmin > tmax {
                    return intersections![];
                }

                intersections![intersection(tmin, *self), intersection(tmax, *self)]
            }
            ShapeType::Cylinder(cmin, cmax, _) => {
                let a = r.direction.x.powi(2) + r.direction.z.powi(2);
                if a.abs() < EPSILON {
                    let mut xs = intersections![];
                    self.intersect_caps(r, &mut xs);
                    return xs;
                }
                let b = 2.0 * r.origin.x * r.direction.x + 2.0 * r.origin.z * r.direction.z;
                let c = r.origin.x.powi(2) + r.origin.z.powi(2) - 1.0;
                let disc = b * b - 4.0 * a * c;
                if disc < 0.0 {
                    return intersections![];
                }
                let mut t0 = (-b - disc.sqrt()) / (2.0 * a);
                let mut t1 = (-b + disc.sqrt()) / (2.0 * a);
                if t0 > t1 {
                    {
                        (t0, t1) = (t1, t0);
                    }
                }
                let mut xs = intersections![];
                let y0 = r.origin.y + t0 * r.direction.y;
                if cmin < y0 && y0 < cmax {
                    xs.push(intersection(t0, *self));
                }
                let y1 = r.origin.y + t1 * r.direction.y;
                if cmin < y1 && y1 < cmax {
                    xs.push(intersection(t1, *self));
                }
                self.intersect_caps(r, &mut xs);
                xs
            }
            ShapeType::Cone(cmin, cmax, _) => {
                let a = r.direction.x.powi(2) - r.direction.y.powi(2) + r.direction.z.powi(2);

                let b = 2.0
                    * (r.origin.x * r.direction.x - r.origin.y * r.direction.y
                        + r.origin.z * r.direction.z);

                let c = r.origin.x.powi(2) - r.origin.y.powi(2) + r.origin.z.powi(2);
                let mut xs = intersections![];
                if a.abs() < EPSILON && !(b.abs() < EPSILON) {
                    let t = c / (-2.0 * b);
                    xs.push(intersection(t, *self));
                } else {
                    let discriminant = b.powi(2) - 4.0 * a * c;

                    if discriminant < 0.0 {
                        return xs;
                    }

                    let double_a = 2.0 * a;
                    let t0 = (-b - discriminant.sqrt()) / double_a;
                    let t1 = (-b + discriminant.sqrt()) / double_a;

                    let y0 = r.origin.y + t0 * r.direction.y;
                    if cmin < y0 && y0 < cmax {
                        xs.push(intersection(t0, *self));
                    }

                    let y1 = r.origin.y + t1 * r.direction.y;
                    if cmin < y1 && y1 < cmax {
                        xs.push(intersection(t1, *self));
                    }
                }
                self.intersect_caps(r, &mut xs);
                xs
            }
        }
    }
    fn local_normal_at(&self, p: Point) -> Vector {
        match self.typ {
            ShapeType::Sphere => p - point(0.0, 0.0, 0.0),
            ShapeType::Plane => vector(0.0, 1.0, 0.0),
            ShapeType::Cube => {
                let maxc = p.x.abs().max(p.y.abs().max(p.z.abs()));
                if maxc == p.x.abs() {
                    return vector(p.x, 0.0, 0.0);
                }
                if maxc == p.y.abs() {
                    return vector(0.0, p.y, 0.0);
                }
                vector(0.0, 0.0, p.z)
            }
            ShapeType::Cylinder(cmin, cmax, _) => {
                let dist = p.x.powi(2) + p.z.powi(2);
                if dist < 1.0 && p.y >= cmax - EPSILON {
                    return vector(0.0, 1.0, 0.0);
                }
                if dist < 1.0 && p.y <= cmin + EPSILON {
                    return vector(0.0, -1.0, 0.0);
                }
                vector(p.x, 0.0, p.z)
            }
            ShapeType::Cone(cmin, cmax, _) => todo!(),
        }
    }
    fn intersect(&self, ray: Ray) -> Vec<Intersection> {
        let r = ray.transform(self.get_transform().inverse());
        self.local_intersect(r)
    }
    fn normal_at(&self, wp: Point) -> Vector {
        let op = self.get_transform().inverse() * wp;
        let on = self.local_normal_at(op);
        let mut wn = self.get_transform().inverse().transposed() * on;
        wn.w = 0.0;
        wn.normalize()
    }
    fn intersect_caps(&self, r: Ray, xs: &mut Vec<Intersection>) {
        match self.typ {
            ShapeType::Cylinder(cmin, cmax, cap) => {
                if !cap || r.direction.y.abs() < EPSILON {
                    return;
                }
                let t = (cmin - r.origin.y) / r.direction.y;
                if r.check_cap_cylinder(t) {
                    xs.push(intersection(t, *self));
                }
                let t = (cmax - r.origin.y) / r.direction.y;
                if r.check_cap_cylinder(t) {
                    xs.push(intersection(t, *self));
                }
            }
            ShapeType::Cone(cmin, cmax, cap) => {
                if !cap || r.direction.y.abs() < EPSILON {
                    return;
                }
                let t = (cmin - r.origin.y) / r.direction.y;
                if r.check_cap_cone(t) {
                    xs.push(intersection(t, *self));
                }
                let t = (cmax - r.origin.y) / r.direction.y;
                if r.check_cap_cone(t) {
                    xs.push(intersection(t, *self));
                }
            }
            _ => panic!("Not supported"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Intersection {
    t: f32,
    object: Shape,
}

fn intersection(t: f32, o: Shape) -> Intersection {
    Intersection { t, object: o }
}

use std::vec as intersections;

fn hit(xs: Vec<Intersection>) -> Option<Intersection> {
    if xs.is_empty() {
        None
    } else {
        let mut min = f32::MAX;
        let mut min_idx = None;
        for (i, x) in xs.iter().enumerate() {
            if x.t > 0.0 && x.t < min {
                min = x.t;
                min_idx = Some(i);
            }
        }
        min_idx.map(|idx| xs[idx])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Light {
    position: Point,
    intensity: Color,
}

fn point_light(position: Point, intensity: Color) -> Light {
    Light {
        position,
        intensity,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Material {
    color: Tuple,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    pattern: Option<Pattern>,
    reflective: f32,
    transparency: f32,
    refractive_index: f32,
}

fn material() -> Material {
    Material {
        color: color(1.0, 1.0, 1.0),
        ambient: 0.1,
        diffuse: 0.9,
        specular: 0.9,
        shininess: 200.0,
        pattern: None,
        reflective: 0.0,
        transparency: 0.0,
        refractive_index: 1.0,
    }
}

fn black() -> Color {
    color(0.0, 0.0, 0.0)
}

fn white() -> Color {
    color(1.0, 1.0, 1.0)
}

fn lighting(
    mat: Material,
    light: Light,
    p: Point,
    eyev: Vector,
    normalv: Vector,
    in_shadow: bool,
) -> Color {
    let col = if let Some(pat) = mat.pattern {
        pat.pattern_at(p)
    } else {
        mat.color
    };
    let eff_color = col * light.intensity;
    let lightv = (light.position - p).normalize();
    let ambient = eff_color * mat.ambient;
    let light_dot_normal = lightv.dot(normalv);
    let diffuse: Color;
    let specular: Color;
    if light_dot_normal < 0.0 {
        diffuse = black();
        specular = black();
    } else {
        diffuse = eff_color * mat.diffuse * light_dot_normal;
        let reflectv = (-lightv).reflect(normalv);
        let reflect_dot_eye = reflectv.dot(eyev);
        if reflect_dot_eye <= 0.0 {
            specular = black();
        } else {
            let factor = reflect_dot_eye.powf(mat.shininess);
            specular = light.intensity * mat.specular * factor;
        }
    }
    if in_shadow {
        return ambient;
    }
    ambient + diffuse + specular
}

#[derive(Debug, Clone, PartialEq)]
struct World {
    light: Option<Light>,
    objects: Vec<Shape>,
}

fn world() -> World {
    World {
        light: None,
        objects: vec![],
    }
}

fn default_world() -> World {
    let mut w = world();
    w.light = Some(point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0)));
    let mut s1 = sphere();
    s1.material.color = color(0.8, 1.0, 0.6);
    s1.material.diffuse = 0.7;
    s1.material.specular = 0.2;
    w.objects.push(s1);
    let mut s2 = sphere();
    s2.set_transform(scaling(0.5, 0.5, 0.5));
    w.objects.push(s2);
    w
}

impl World {
    fn intersect(&self, r: Ray) -> Vec<Intersection> {
        let mut v: Vec<Intersection> = self.objects.iter().flat_map(|o| o.intersect(r)).collect();
        v.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        v
    }
    fn shade_hit(&self, comps: Computations, remaining: u32) -> Color {
        let surface = lighting(
            comps.object.material,
            self.light.unwrap(),
            comps.over_point,
            comps.eyev,
            comps.normalv,
            self.is_shadowed(comps.over_point),
        );
        let reflected = self.reflected_color(comps, remaining);
        let refracted = self.refracted_color(comps, remaining);
        let mat = comps.object.material;
        if mat.reflective > 0.0 && mat.transparency > 0.0 {
            let reflectance = comps.schlick();
            surface + reflected * reflectance + refracted * (1.0 - reflectance)
        } else {
            surface + reflected + refracted
        }
    }
    fn color_at(&self, r: Ray, remaining: u32) -> Color {
        let xs = self.intersect(r);
        if let Some(x) = hit(xs.clone()) {
            let comps = x.prepare_computations(r, &xs);
            self.shade_hit(comps, remaining)
        } else {
            black()
        }
    }
    fn is_shadowed(&self, p: Point) -> bool {
        let v = self.light.unwrap().position - p;
        let distance = v.magnitude();
        let direction = v.normalize();
        let r = ray(p, direction);
        let xs = self.intersect(r);
        if let Some(h) = hit(xs) {
            if h.t < distance {
                return true;
            }
        }
        false
    }
    fn reflected_color(&self, comps: Computations, remaining: u32) -> Color {
        if remaining == 0 {
            return color(0.0, 0.0, 0.0);
        }
        if comps.object.get_material().reflective == 0.0 {
            return black();
        }
        let reflect_ray = ray(comps.over_point, comps.reflectv);
        let col = self.color_at(reflect_ray, remaining - 1);
        col * comps.object.get_material().reflective
    }
    fn refracted_color(&self, comps: Computations, remaining: u32) -> Color {
        if remaining == 0 {
            return black();
        }
        if comps.object.get_material().transparency == 0.0 {
            return black();
        }
        let n_ratio = comps.n1 / comps.n2;
        let cos_i = comps.eyev.dot(comps.normalv);
        let sin2_t = n_ratio.powi(2) * (1.0 - cos_i.powi(2));
        if sin2_t > 1.0 {
            return black();
        }
        let cos_t = (1.0 - sin2_t).sqrt();
        let direction = comps.normalv * (n_ratio * cos_i - cos_t) - comps.eyev * n_ratio;
        let refract_ray = ray(comps.under_point, direction);
        self.color_at(refract_ray, remaining - 1) * comps.object.material.transparency
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Computations {
    t: f32,
    object: Shape,
    point: Point,
    eyev: Vector,
    normalv: Vector,
    inside: bool,
    over_point: Point,
    reflectv: Vector,
    n1: f32,
    n2: f32,
    under_point: Point,
}

impl Computations {
    fn schlick(&self) -> f32 {
        let mut cos = self.eyev.dot(self.normalv);
        if self.n1 > self.n2 {
            let n = self.n1 / self.n2;
            let sin2_t = n.powi(2) * (1.0 - cos.powi(2));
            if sin2_t > 1.0 {
                return 1.0;
            }
            let cos_t = (1.0 - sin2_t).sqrt();
            cos = cos_t;
        }
        let r0 = ((self.n1 - self.n2) / (self.n1 + self.n2)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cos).powi(5)
    }
}

impl Intersection {
    fn prepare_computations(&self, r: Ray, xs: &Vec<Intersection>) -> Computations {
        let p = r.position(self.t);
        let mut comps = Computations {
            t: self.t,
            object: self.object,
            point: p,
            eyev: -r.direction,
            normalv: self.object.normal_at(p),
            inside: false,
            over_point: point(0.0, 0.0, 0.0),
            reflectv: vector(0.0, 0.0, 0.0),
            n1: 1.0,
            n2: 1.0,
            under_point: point(0.0, 0.0, 0.0),
        };
        let mut containers: Vec<Shape> = Vec::new();
        for i in xs.iter() {
            if i == self {
                if containers.is_empty() {
                    comps.n1 = 1.0;
                } else {
                    comps.n1 = containers.last().unwrap().get_material().refractive_index;
                }
            }
            if containers.contains(&i.object) {
                let idx = containers.iter().position(|&x| x == i.object).unwrap();
                containers.remove(idx);
            } else {
                containers.push(i.object);
            }
            if i == self {
                if containers.is_empty() {
                    comps.n2 = 1.0;
                } else {
                    comps.n2 = containers.last().unwrap().get_material().refractive_index;
                }
                break;
            }
        }
        if comps.normalv.dot(comps.eyev) < 0.0 {
            comps.inside = true;
            comps.normalv = -comps.normalv
        }
        comps.over_point = comps.point + comps.normalv * EPSILON;
        comps.under_point = comps.point - comps.normalv * EPSILON;
        comps.reflectv = r.direction.reflect(comps.normalv);
        comps
    }
}

fn view_transform(from: Point, to: Point, up: Vector) -> Matrix4x4<f32> {
    let forward = (to - from).normalize();
    let upn = up.normalize();
    let left = forward.cross(upn);
    let true_up = left.cross(forward);
    let orientation = Matrix4x4::<f32> {
        data: [
            left.x, left.y, left.z, 0.0, true_up.x, true_up.y, true_up.z, 0.0, -forward.x,
            -forward.y, -forward.z, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };
    orientation * translation(-from.x, -from.y, -from.z)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Camera {
    hsize: usize,
    vsize: usize,
    field_of_view: f32,
    transform: Matrix4x4<f32>,
    half_width: f32,
    half_height: f32,
    pixel_size: f32,
}

fn camera(hsize: usize, vsize: usize, field_of_view: f32) -> Camera {
    let mut c = Camera {
        hsize,
        vsize,
        field_of_view,
        transform: identity_matrix(),
        half_width: 0.0,
        half_height: 0.0,
        pixel_size: 0.0,
    };
    let half_view = (c.field_of_view / 2.0).tan();
    let aspect = c.hsize as f32 / c.vsize as f32;
    if aspect >= 1.0 {
        c.half_width = half_view;
        c.half_height = half_view / aspect;
    } else {
        c.half_width = half_view * aspect;
        c.half_height = half_view;
    }
    c.pixel_size = (c.half_width * 2.0) / c.hsize as f32;
    c
}

impl Camera {
    fn ray_for_pixel(&self, px: usize, py: usize) -> Ray {
        let xoffset = (px as f32 + 0.5) * self.pixel_size;
        let yoffset = (py as f32 + 0.5) * self.pixel_size;
        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;
        let pixel = self.transform.inverse() * point(world_x, world_y, -1.0);
        let origin = self.transform.inverse() * point(0.0, 0.0, 0.0);
        let direction = (pixel - origin).normalize();
        Ray { origin, direction }
    }
}

fn render(c: Camera, w: World) -> Canvas {
    let mut image = canvas(c.hsize, c.vsize);
    for y in 0..c.vsize {
        for x in 0..c.hsize {
            let r = c.ray_for_pixel(x, y);
            let color = w.color_at(r, 5);
            image.write_pixel(x, y, color);
        }
    }
    image
}

fn plane() -> Shape {
    Shape {
        typ: ShapeType::Plane,
        transform: identity_matrix(),
        material: material(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PatType {
    Stripe,
    Gradient,
    Ring,
    Checker,
    Test,
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct Pattern {
    typ: PatType,
    a: Color,
    b: Color,
    transform: Matrix4x4<f32>,
}

fn stripe_pattern(a: Color, b: Color) -> Pattern {
    Pattern {
        typ: PatType::Stripe,
        a,
        b,
        transform: identity_matrix(),
    }
}

fn gradient_pattern(a: Color, b: Color) -> Pattern {
    Pattern {
        typ: PatType::Gradient,
        a,
        b,
        transform: identity_matrix(),
    }
}

fn ring_pattern(a: Color, b: Color) -> Pattern {
    Pattern {
        typ: PatType::Ring,
        a,
        b,
        transform: identity_matrix(),
    }
}

fn checkers_pattern(a: Color, b: Color) -> Pattern {
    Pattern {
        typ: PatType::Checker,
        a,
        b,
        transform: identity_matrix(),
    }
}

impl Pattern {
    fn pattern_at(&self, p: Point) -> Color {
        match self.typ {
            PatType::Stripe => {
                if p.x.floor() as i64 % 2 == 0 {
                    self.a
                } else {
                    self.b
                }
            }
            PatType::Gradient => {
                let distance = self.b - self.a;
                let fraction = p.x - p.x.floor();
                self.a + distance * fraction
            }
            PatType::Ring => {
                if (p.x * p.x + p.z * p.z).sqrt() as i64 % 2 == 0 {
                    self.a
                } else {
                    self.b
                }
            }
            PatType::Checker => {
                if (p.x.trunc() + p.y.trunc() + p.z.trunc()) as i64 % 2 == 0 {
                    self.a
                } else {
                    self.b
                }
            }
            PatType::Test => color(p.x, p.y, p.z),
        }
    }
    fn stripe_at_object(&self, obj: Shape, p: Point) -> Color {
        let transform = obj.get_transform();
        let op = transform.inverse() * p;
        let pp = self.transform.inverse() * op;
        self.pattern_at(pp)
    }
}

fn cube() -> Shape {
    Shape {
        typ: ShapeType::Cube,
        transform: identity_matrix(),
        material: material(),
    }
}

fn cylinder() -> Shape {
    Shape {
        typ: ShapeType::Cylinder(-f32::INFINITY, f32::INFINITY, false),
        transform: identity_matrix(),
        material: material(),
    }
}

fn cone() -> Shape {
    Shape {
        typ: ShapeType::Cone(-f32::INFINITY, f32::INFINITY, false),
        transform: identity_matrix(),
        material: material(),
    }
}

fn main() {
    use std::f32::consts::PI;

    let mut floor = plane();
    floor.material = material();
    floor.material.color = color(1.0, 0.9, 0.9);
    floor.material.specular = 0.0;
    floor.material.pattern = Some(stripe_pattern(white(), black()));
    floor.material.reflective = 0.5;

    let mut left_wall = plane();
    left_wall.transform = translation(0.0, 0.0, 5.0) * rotation_y(-PI / 4.0) * rotation_x(PI / 2.0);
    left_wall.material = floor.material;
    left_wall.material.reflective = 0.0;
    left_wall.material.pattern = None;

    let mut right_wall = plane();
    right_wall.transform = translation(0.0, 0.0, 5.0) * rotation_y(PI / 4.0) * rotation_x(PI / 2.0);
    right_wall.material = floor.material;
    right_wall.material.reflective = 0.0;
    right_wall.material.pattern = None;

    let mut middle = glass_sphere();
    middle.transform = translation(-0.5, 1.0, 0.5);
    middle.material.reflective = 1.0;

    let mut right = cylinder();
    right.typ = ShapeType::Cylinder(0.0, 1.0, true);
    right.transform = translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.5);
    right.material.reflective = 1.0;

    let mut left = cube();
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33);
    left.material = material();
    left.material.color = color(1.0, 0.8, 0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;

    let mut w = world();
    w.light = Some(point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0)));
    w.objects = vec![floor, left_wall, right_wall, middle, right, left];
    let mut cam = camera(1000, 500, PI / 3.0);
    cam.transform = view_transform(
        point(0.0, 1.5, -5.0),
        point(0.0, 1.0, 0.0),
        vector(0.0, 1.0, 0.0),
    );
    let canvas = render(cam, w);
    println!("{}", canvas.to_ppm());
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::{
        black, camera, canvas, checkers_pattern, color, cone, cube, cylinder, default_world,
        glass_sphere, gradient_pattern, hit, identity_matrix, intersection, intersections,
        lighting, material, plane, point, point_light, ray, render, ring_pattern, rotation_x,
        rotation_y, rotation_z, scaling, shearing, sphere, stripe_pattern, translation, tuple,
        vector, view_transform, white, world, Intersection, Matrix2x2, Matrix3x3, Matrix4x4,
        Pattern, ShapeType, Tuple, EPSILON,
    };

    #[test]
    fn is_point() {
        let p = point(4.3, -4.2, 3.1);
        assert!(p.is_point());
    }

    #[test]
    fn is_vector() {
        let v = vector(4.3, -4.2, 3.1);
        assert!(v.is_vector());
    }

    #[test]
    fn add_tuples() {
        let a1 = tuple(3.0, -2.0, 5.0, 1.0);
        let a2 = tuple(-2.0, 3.0, 1.0, 0.0);
        assert_eq!(a1 + a2, tuple(1.0, 1.0, 6.0, 1.0));
    }

    #[test]
    fn sub_points() {
        let p1 = point(3.0, 2.0, 1.0);
        let p2 = point(5.0, 6.0, 7.0);
        assert_eq!(p1 - p2, vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_vec_from_point() {
        let p = point(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);
        assert_eq!(p - v, point(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_vectors() {
        let v1 = vector(3.0, 2.0, 1.0);
        let v2 = vector(5.0, 6.0, 7.0);
        assert_eq!(v1 - v2, vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn neg_tuple() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(-a, tuple(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn mul_tuple_by_scalar() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(a * 3.5, tuple(3.5, -7.0, 10.5, -14.0));
    }

    #[test]
    fn div_tuple_by_scalar() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(a / 2.0, tuple(0.5, -1.0, 1.5, -2.0));
    }

    #[test]
    fn vector_magnitude() {
        assert_eq!(1.0, vector(1.0, 0.0, 0.0).magnitude());
        assert_eq!(1.0, vector(0.0, 1.0, 0.0).magnitude());
        assert_eq!(1.0, vector(0.0, 0.0, 1.0).magnitude());
        assert_eq!((14.0f32).sqrt(), vector(1.0, 2.0, 3.0).magnitude());
        assert_eq!((14.0f32).sqrt(), vector(-1.0, -2.0, -3.0).magnitude());
    }

    #[test]
    fn normalize_vector() {
        assert_eq!(vector(1.0, 0.0, 0.0), vector(4.0, 0.0, 0.0).normalize());
        assert_eq!(
            vector(
                1.0 / 14.0f32.sqrt(),
                2.0 / 14.0f32.sqrt(),
                3.0 / 14.0f32.sqrt()
            ),
            vector(1.0, 2.0, 3.0).normalize()
        );
        assert!((1.0 - vector(1.0, 2.0, 3.0).normalize().magnitude()).abs() < EPSILON);
    }

    #[test]
    fn dot_product() {
        assert_eq!(20.0, vector(1.0, 2.0, 3.0).dot(vector(2.0, 3.0, 4.0)));
    }

    #[test]
    fn cross_product() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);
        assert_eq!(vector(-1.0, 2.0, -1.0), a.cross(b));
        assert_eq!(vector(1.0, -2.0, 1.0), b.cross(a));
    }

    #[test]
    fn multiply_colors() {
        assert_eq!(
            vector(0.9, 0.2, 0.04),
            vector(1.0, 0.2, 0.4) * vector(0.9, 1.0, 0.1)
        );
    }

    #[test]
    fn writing_pixel() {
        let mut c = canvas(10, 20);
        let red = color(1.0, 0.0, 0.0);
        c.write_pixel(2, 3, red);
        assert_eq!(red, c.pixel_at(2, 3));
    }

    #[test]
    fn constructing_ppm() {
        let mut c = canvas(5, 3);
        let c1 = color(1.5, 0.0, 0.0);
        let c2 = color(0.0, 0.5, 0.0);
        let c3 = color(-0.5, 0.0, 1.0);
        c.write_pixel(0, 0, c1);
        c.write_pixel(2, 1, c2);
        c.write_pixel(4, 2, c3);
        let ppm = c.to_ppm();
        assert_eq!(
            ppm,
            "P3
5 3
255
255 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 128 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 255
"
        );
    }

    #[test]
    fn constructing_matrix() {
        let m = Matrix4x4::<f32> {
            data: [
                1.0, 2.0, 3.0, 4.0, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 11.0, 12.0, 13.5, 14.5, 15.5,
                16.5,
            ],
        };
        assert_eq!(1.0, m.at(0, 0));
        assert_eq!(4.0, m.at(0, 3));
        assert_eq!(5.5, m.at(1, 0));
        assert_eq!(7.5, m.at(1, 2));
        assert_eq!(11.0, m.at(2, 2));
        assert_eq!(13.5, m.at(3, 0));
        assert_eq!(15.5, m.at(3, 2));
    }

    #[test]
    fn compare_matrices() {
        let a = Matrix2x2::<f32> {
            data: [1.0, 2.0, 3.0, 4.0],
        };
        let b = Matrix2x2::<f32> {
            data: [-1.0, 2.0, -3.0, 4.0],
        };
        assert_eq!(a, a);
        assert_ne!(a, b);
    }

    #[test]
    fn multiply_matrices() {
        let a = Matrix4x4::<f32> {
            data: [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            ],
        };
        let b = Matrix4x4::<f32> {
            data: [
                -2.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, -1.0, 4.0, 3.0, 6.0, 5.0, 1.0, 2.0, 7.0, 8.0,
            ],
        };
        let c = Matrix4x4::<f32> {
            data: [
                20.0, 22.0, 50.0, 48.0, 44.0, 54.0, 114.0, 108.0, 40.0, 58.0, 110.0, 102.0, 16.0,
                26.0, 46.0, 42.0,
            ],
        };
        assert_eq!(c, a * b);
    }

    #[test]
    fn mul_matrix_by_tuple() {
        let a = Matrix4x4::<f32> {
            data: [
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 4.0, 2.0, 8.0, 6.0, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };
        let b = tuple(1.0, 2.0, 3.0, 1.0);
        assert_eq!(tuple(18.0, 24.0, 33.0, 1.0), a * b);
    }

    #[test]
    fn mul_by_identity_matrix() {
        let a = Matrix4x4::<f32> {
            data: [
                0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 16.0, 4.0, 8.0, 16.0, 32.0,
            ],
        };
        assert_eq!(a.clone(), a * identity_matrix());
        let b = tuple(1.0, 2.0, 3.0, 4.0);
        assert_eq!(identity_matrix() * b, b);
    }

    #[test]
    fn matrix_transpose() {
        let mut a = Matrix4x4::<f32> {
            data: [
                0.0, 9.0, 3.0, 0.0, 9.0, 8.0, 0.0, 8.0, 1.0, 8.0, 5.0, 3.0, 0.0, 0.0, 5.0, 8.0,
            ],
        };
        let b = Matrix4x4::<f32> {
            data: [
                0.0, 9.0, 1.0, 0.0, 9.0, 8.0, 8.0, 0.0, 3.0, 0.0, 5.0, 5.0, 0.0, 8.0, 3.0, 8.0,
            ],
        };
        assert_eq!(a.clone().transposed(), b);
        a.transpose();
        assert_eq!(a, b);
    }

    #[test]
    fn determinant2x2() {
        let a = Matrix2x2::<f32> {
            data: [1.0, 5.0, -3.0, 2.0],
        };
        assert_eq!(17.0, a.determinant());
    }

    #[test]
    fn submatrices() {
        let a = Matrix3x3::<f32> {
            data: [1.0, 5.0, 0.0, -3.0, 2.0, 7.0, 0.0, 6.0, -3.0],
        };
        let b = Matrix2x2::<f32> {
            data: [-3.0, 2.0, 0.0, 6.0],
        };
        let c = Matrix4x4::<f32> {
            data: [
                -6.0, 1.0, 1.0, 6.0, -8.0, 5.0, 8.0, 6.0, -1.0, 0.0, 8.0, 2.0, -7.0, 1.0, -1.0, 1.0,
            ],
        };
        let d = Matrix3x3::<f32> {
            data: [-6.0, 1.0, 6.0, -8.0, 8.0, 6.0, -7.0, -1.0, 1.0],
        };
        assert_eq!(a.submatrix(0, 2), b);
        assert_eq!(c.submatrix(2, 1), d);
    }

    #[test]
    fn minor() {
        let a = Matrix3x3::<f32> {
            data: [3.0, 5.0, 0.0, 2.0, -1.0, -7.0, 6.0, -1.0, 5.0],
        };
        assert_eq!(25.0, a.minor(1, 0));
    }

    #[test]
    fn cofactor() {
        let a = Matrix3x3::<f32> {
            data: [3.0, 5.0, 0.0, 2.0, -1.0, -7.0, 6.0, -1.0, 5.0],
        };
        assert_eq!(-12.0, a.cofactor(0, 0));
        assert_eq!(-25.0, a.cofactor(1, 0));
    }

    #[test]
    fn determinant3x3and4x4() {
        let a = Matrix3x3::<f32> {
            data: [1.0, 2.0, 6.0, -5.0, 8.0, -4.0, 2.0, 6.0, 4.0],
        };
        assert_eq!(-196.0, a.determinant());
        let b = Matrix4x4::<f32> {
            data: [
                -2.0, -8.0, 3.0, 5.0, -3.0, 1.0, 7.0, 3.0, 1.0, 2.0, -9.0, 6.0, -6.0, 7.0, 7.0,
                -9.0,
            ],
        };
        assert_eq!(-4071.0, b.determinant());
    }

    #[test]
    fn inverse() {
        let a = Matrix4x4::<f32> {
            data: [
                -5.0, 2.0, 6.0, -8.0, 1.0, -5.0, 1.0, 8.0, 7.0, 7.0, -6.0, -7.0, 1.0, -3.0, 7.0,
                4.0,
            ],
        };
        let b = Matrix4x4::<f32> {
            data: [
                0.21804512,
                0.45112783,
                0.24060151,
                -0.04511278,
                -0.8082707,
                -1.456767,
                -0.44360903,
                0.5206767,
                -0.078947365,
                -0.2236842,
                -0.05263158,
                0.19736843,
                -0.52255636,
                -0.81390977,
                -0.30075186,
                0.30639097,
            ],
        };
        assert_eq!(b, a.inverse());
    }

    #[test]
    fn mul_by_inverse() {
        let a = Matrix4x4::<f32> {
            data: [
                3.0, -9.0, 7.0, 3.0, 3.0, -8.0, 2.0, -9.0, -4.0, 4.0, 4.0, 1.0, -6.0, 5.0, -1.0,
                1.0,
            ],
        };
        let b = Matrix4x4::<f32> {
            data: [
                8.0, 2.0, 2.0, 2.0, 3.0, -1.0, 7.0, 0.0, 7.0, 0.0, 5.0, 4.0, 6.0, -2.0, 0.0, 5.0,
            ],
        };
        let c = a.clone() * b.clone();
        let d = Matrix4x4::<f32> {
            data: [
                3.0, -8.999999, 7.0, 3.0, 3.0, -7.999999, 2.0, -9.0, -4.0, 4.0, 4.0000005,
                0.99999976, -6.0, 5.0, -0.9999995, 0.9999999,
            ],
        };
        assert_eq!(d, c * b.inverse());
    }

    #[test]
    fn translate_point() {
        let transform = translation(5.0, -3.0, 2.0);
        let inv = transform.inverse();
        let p = point(-3.0, 4.0, 5.0);
        assert_eq!(point(2.0, 1.0, 7.0), transform * p);
        assert_eq!(point(-8.0, 7.0, 3.0), inv * p);
    }

    #[test]
    fn translate_vector() {
        let transform = translation(5.0, -3.0, 2.0);
        let v = vector(-3.0, 4.0, 5.0);
        assert_eq!(v, transform * v);
    }

    #[test]
    fn scale() {
        let transform = scaling(2.0, 3.0, 4.0);
        let p = point(-4.0, 6.0, 8.0);
        assert_eq!(point(-8.0, 18.0, 32.0), transform.clone() * p);
        let v = vector(-4.0, 6.0, 8.0);
        assert_eq!(vector(-8.0, 18.0, 32.0), transform.clone() * v);
        let inv = transform.inverse();
        assert_eq!(vector(-2.0, 2.0, 2.0), inv * v);
        let transform2 = scaling(-1.0, 1.0, 1.0);
        assert_eq!(point(-2.0, 3.0, 4.0), transform2 * point(2.0, 3.0, 4.0));
    }

    #[test]
    fn rotations() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_x(PI / 4.0);
        let full_quarter = rotation_x(PI / 2.0);
        assert_eq!(
            point(0.0, 2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
            half_quarter.clone() * p
        );
        assert_eq!(point(0.0, 0.0, 1.0), full_quarter * p);
        let inv = half_quarter.inverse();
        assert_eq!(point(0.0, 0.7071068, -0.7071068), inv * p);
        let p = point(0.0, 0.0, 1.0);
        let half_quarter = rotation_y(PI / 4.0);
        let full_quarter = rotation_y(PI / 2.0);
        assert_eq!(
            point(2f32.sqrt() / 2.0, 0.0, 2f32.sqrt() / 2.0),
            half_quarter.clone() * p
        );
        assert_eq!(point(1.0, 0.0, 0.0), full_quarter * p);
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_z(PI / 4.0);
        let full_quarter = rotation_z(PI / 2.0);
        assert_eq!(
            point(-2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0, 0.0),
            half_quarter.clone() * p
        );
        assert_eq!(point(-1.0, 0.0, 0.0), full_quarter * p);
    }

    #[test]
    fn shear() {
        let transform = shearing(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(point(5.0, 3.0, 4.0), transform * p);
        let transform = shearing(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(point(6.0, 3.0, 4.0), transform * p);
        let transform = shearing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(point(2.0, 5.0, 4.0), transform * p);
        let transform = shearing(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(point(2.0, 7.0, 4.0), transform * p);
        let transform = shearing(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(point(2.0, 3.0, 6.0), transform * p);
        let transform = shearing(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(point(2.0, 3.0, 7.0), transform * p);
    }

    #[test]
    fn sequence() {
        let p = point(1.0, 0.0, 1.0);
        let a = rotation_x(PI / 2.0);
        let b = scaling(5.0, 5.0, 5.0);
        let c = translation(10.0, 5.0, 7.0);
        let p2 = a * p;
        assert_eq!(p2, point(1.0, -1.0, 0.0));
        let p3 = b * p2;
        assert_eq!(p3, point(5.0, -5.0, 0.0));
        let p4 = c * p3;
        assert_eq!(p4, point(15.0, 0.0, 7.0));
    }

    #[test]
    fn ray_creation() {
        let origin = point(1.0, 2.0, 3.0);
        let direction = vector(4.0, 5.0, 6.0);
        let r = ray(origin, direction);
        assert_eq!(origin, r.origin);
        assert_eq!(direction, r.direction);
    }

    #[test]
    fn ray_position() {
        let r = ray(point(2.0, 3.0, 4.0), vector(1.0, 0.0, 0.0));
        assert_eq!(r.position(0.0), point(2.0, 3.0, 4.0));
        assert_eq!(r.position(1.0), point(3.0, 3.0, 4.0));
        assert_eq!(r.position(-1.0), point(1.0, 3.0, 4.0));
        assert_eq!(r.position(2.5), point(4.5, 3.0, 4.0));
    }

    #[test]
    fn ray_sphere_intersections() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere();
        let xs = s.intersect(r);
        if xs.len() == 2 {
            assert_eq!(xs[0].t, 4.0);
            assert_eq!(xs[1].t, 6.0);
        } else {
            assert!(false);
        }

        let r = ray(point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere();
        let xs = s.intersect(r);
        if xs.len() == 2 {
            assert_eq!(xs[0].t, 5.0);
            assert_eq!(xs[1].t, 5.0);
        } else {
            assert!(false);
        }

        let r = ray(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere();
        let xs = s.intersect(r);
        assert_eq!(xs.len(), 0);

        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let s = sphere();
        let xs = s.intersect(r);
        if xs.len() == 2 {
            assert_eq!(xs[0].t, -1.0);
            assert_eq!(xs[1].t, 1.0);
        } else {
            assert!(false);
        }

        let r = ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
        let s = sphere();
        let xs = s.intersect(r);
        if xs.len() == 2 {
            assert_eq!(xs[0].t, -6.0);
            assert_eq!(xs[1].t, -4.0);
        } else {
            assert!(false);
        }

        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere();
        let xs = s.intersect(r);
        if xs.len() == 2 {
            assert_eq!(xs[0].object, s);
            assert_eq!(xs[1].object, s);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn aggregate_intersections() {
        let s = sphere();
        let i = intersection(3.5, s);
        assert_eq!(i.t, 3.5);
        assert_eq!(i.object, s);

        let s = sphere();
        let i1 = intersection(1.0, s);
        let i2 = intersection(2.0, s);
        let xs = intersections![i1, i2];
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[1].t, 2.0);
    }

    #[test]
    fn hits() {
        let s = sphere();
        let i1 = intersection(1.0, s);
        let i2 = intersection(2.0, s);
        let xs = intersections![i1, i2];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i1);
        } else {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(-1.0, s);
        let i2 = intersection(1.0, s);
        let xs = intersections![i1, i2];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i2);
        } else {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(-2.0, s);
        let i2 = intersection(-1.0, s);
        let xs = intersections![i1, i2];
        if let Some(_) = hit(xs) {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(5.0, s);
        let i2 = intersection(7.0, s);
        let i3 = intersection(-3.0, s);
        let i4 = intersection(2.0, s);
        let xs = intersections![i1, i2, i3, i4];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i4);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn translating_ray() {
        let r = ray(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0));
        let m = translation(3.0, 4.0, 5.0);
        let r2 = r.transform(m);
        assert_eq!(r2.origin, point(4.0, 6.0, 8.0));
        assert_eq!(r2.direction, vector(0.0, 1.0, 0.0));
    }

    #[test]
    fn scaling_ray() {
        let r = ray(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0));
        let m = scaling(2.0, 3.0, 4.0);
        let r2 = r.transform(m);
        assert_eq!(r2.origin, point(2.0, 6.0, 12.0));
        assert_eq!(r2.direction, vector(0.0, 3.0, 0.0));
    }

    #[test]
    fn sphere_set_transform() {
        let mut s = sphere();
        let t = translation(2.0, 3.0, 4.0);
        s.set_transform(t);
        assert_eq!(s.transform, t);
    }

    #[test]
    fn intersect_scaled_sphere_with_ray() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut s = sphere();
        s.set_transform(scaling(2.0, 2.0, 2.0));
        let xs = s.intersect(r);
        if xs.len() == 2 {
            assert_eq!(xs[0].t, 3.0);
            assert_eq!(xs[1].t, 7.0);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn intersect_translated_sphere_with_ray() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut s = sphere();
        s.set_transform(translation(5.0, 0.0, 0.0));
        let xs = s.intersect(r);
        if xs.len() == 0 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn sphere_normals() {
        let s = sphere();
        let n = s.normal_at(point(1.0, 0.0, 0.0));
        assert_eq!(n, vector(1.0, 0.0, 0.0));

        let s = sphere();
        let n = s.normal_at(point(0.0, 1.0, 0.0));
        assert_eq!(n, vector(0.0, 1.0, 0.0));

        let s = sphere();
        let n = s.normal_at(point(0.0, 0.0, 1.0));
        assert_eq!(n, vector(0.0, 0.0, 1.0));

        let s = sphere();
        let n = s.normal_at(point(
            3.0f32.sqrt() / 3.0,
            3.0f32.sqrt() / 3.0,
            3.0f32.sqrt() / 3.0,
        ));
        assert_eq!(n, vector(0.5773503, 0.5773503, 0.5773503));

        let s = sphere();
        let n = s.normal_at(point(
            3.0f32.sqrt() / 3.0,
            3.0f32.sqrt() / 3.0,
            3.0f32.sqrt() / 3.0,
        ));
        assert_eq!(n, n.normalize().normalize());
    }

    #[test]
    fn normals_translated() {
        let mut s = sphere();
        s.set_transform(translation(0.0, 1.0, 0.0));
        let n = s.normal_at(point(0.0, 1.70711, -0.70711));
        assert_eq!(n, vector(0.0, 0.7071068, -0.70710677));

        let mut s = sphere();
        let m = scaling(1.0, 0.5, 1.0) * rotation_z(PI / 5.0);
        s.set_transform(m);
        let n = s.normal_at(point(0.0, 2f32.sqrt() / 2.0, -2f32.sqrt() / 2.0));
        assert_eq!(n, vector(0.0, 0.97014254, -0.24253564));
    }

    #[test]
    fn vector_reflection() {
        let v = vector(1.0, -1.0, 0.0);
        let n = vector(0.0, 1.0, 0.0);
        let r = v.reflect(n);
        assert_eq!(r, vector(1.0, 1.0, 0.0));

        let v = vector(0.0, -1.0, 0.0);
        let n = vector(2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0, 0.0);
        let r = v.reflect(n);
        assert_eq!(r, vector(1.0, 0.0, 0.0));
    }

    #[test]
    fn light_creation() {
        let intensity = color(1.0, 1.0, 1.0);
        let position = point(0.0, 0.0, 0.0);
        let light = point_light(position, intensity);
        assert_eq!(light.position, position);
        assert_eq!(light.intensity, intensity);
    }

    #[test]
    fn material_creation() {
        let m = material();
        assert_eq!(m.color, color(1.0, 1.0, 1.0));
        assert_eq!(m.ambient, 0.1);
        assert_eq!(m.diffuse, 0.9);
        assert_eq!(m.specular, 0.9);
        assert_eq!(m.shininess, 200.0);
    }

    #[test]
    fn sphere_material() {
        let s = sphere();
        assert_eq!(s.material, material());

        let mut s = sphere();
        let mut m = material();
        m.ambient = 1.0;
        s.material = m;
        assert_eq!(s.material, m);
    }

    #[test]
    fn lighting_with_the_eye() {
        let m = material();
        let position = point(0.0, 0.0, 0.0);
        let eyev = vector(0.0, 0.0, -1.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0));
        let result = lighting(m, light, position, eyev, normalv, false);
        assert_eq!(result, color(1.9, 1.9, 1.9));

        let eyev = vector(0.0, 2f32.sqrt() / 2.0, -2f32.sqrt() / 2.0);
        let result = lighting(m, light, position, eyev, normalv, false);
        assert_eq!(result, color(1.0, 1.0, 1.0));

        let eyev = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 10.0, -10.0), color(1.0, 1.0, 1.0));
        let result = lighting(m, light, position, eyev, normalv, false);
        assert_eq!(result, color(0.7363961, 0.7363961, 0.7363961));

        let eyev = vector(0.0, -2f32.sqrt() / 2.0, -2f32.sqrt() / 2.0);
        let result = lighting(m, light, position, eyev, normalv, false);
        assert_eq!(result, color(1.6363853, 1.6363853, 1.6363853));

        let eyev = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, 10.0), color(1.0, 1.0, 1.0));
        let result = lighting(m, light, position, eyev, normalv, false);
        assert_eq!(result, color(0.1, 0.1, 0.1));
    }

    #[test]
    fn world_intersect() {
        let w = default_world();
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let xs = w.intersect(r);
        if xs.len() == 4 {
            assert_eq!(xs[0].t, 4.0);
            assert_eq!(xs[1].t, 4.5);
            assert_eq!(xs[2].t, 5.5);
            assert_eq!(xs[3].t, 6.0);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn precompute_intersection() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let shape = sphere();
        let i = intersection(4.0, shape);
        let comps = i.prepare_computations(r, &vec![i]);
        assert_eq!(comps.t, i.t);
        assert_eq!(comps.object, i.object);
        assert_eq!(comps.point, point(0.0, 0.0, -1.0));
        assert_eq!(comps.eyev, vector(0.0, 0.0, -1.0));
        assert_eq!(comps.normalv, vector(0.0, 0.0, -1.0));
    }

    #[test]
    fn inside_outside_intersections() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let shape = sphere();
        let i = intersection(4.0, shape);
        let comps = i.prepare_computations(r, &vec![i]);
        assert_eq!(comps.inside, false);

        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let shape = sphere();
        let i = intersection(1.0, shape);
        let comps = i.prepare_computations(r, &vec![i]);
        assert_eq!(comps.point, point(0.0, 0.0, 1.0));
        assert_eq!(comps.eyev, vector(0.0, 0.0, -1.0));
        assert_eq!(comps.inside, true);
        assert_eq!(comps.normalv, vector(0.0, 0.0, -1.0));
    }

    #[test]
    fn shading_intersection() {
        let w = default_world();
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let shape = w.objects[0];
        let i = intersection(4.0, shape);
        let comps = i.prepare_computations(r, &vec![i]);
        let c = w.shade_hit(comps, 5);
        assert_eq!(c, color(0.38066125, 0.4758265, 0.28549594));

        let mut w = default_world();
        w.light = Some(point_light(point(0.0, 0.25, 0.0), color(1.0, 1.0, 1.0)));
        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let shape = w.objects[1];
        let i = intersection(0.5, shape);
        let comps = i.prepare_computations(r, &vec![i]);
        let c = w.shade_hit(comps, 5);
        assert_eq!(c, color(0.9049845, 0.9049845, 0.9049845));
    }

    #[test]
    fn color_at() {
        let w = default_world();
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));
        let c = w.color_at(r, 5);
        assert_eq!(c, color(0.0, 0.0, 0.0));

        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let c = w.color_at(r, 5);
        assert_eq!(c, color(0.38066125, 0.4758265, 0.28549594));

        let mut w = default_world();
        w.objects[0].material.ambient = 1.0;
        w.objects[1].material.ambient = 1.0;
        let r = ray(point(0.0, 0.0, 0.75), vector(0.0, 0.0, -1.0));
        let c = w.color_at(r, 5);
        assert_eq!(c, w.objects[1].material.color);
    }

    #[test]
    fn view_transforms() {
        let from = point(0.0, 0.0, 0.0);
        let to = point(0.0, 0.0, -1.0);
        let up = vector(0.0, 1.0, 0.0);
        let t = view_transform(from, to, up);
        assert_eq!(t, identity_matrix());

        let from = point(0.0, 0.0, 0.0);
        let to = point(0.0, 0.0, 1.0);
        let up = vector(0.0, 1.0, 0.0);
        let t = view_transform(from, to, up);
        assert_eq!(t, scaling(-1.0, 1.0, -1.0));

        let from = point(0.0, 0.0, 8.0);
        let to = point(0.0, 0.0, 0.0);
        let up = vector(0.0, 1.0, 0.0);
        let t = view_transform(from, to, up);
        assert_eq!(t, translation(0.0, 0.0, -8.0));

        let from = point(1.0, 3.0, 2.0);
        let to = point(4.0, -2.0, 8.0);
        let up = vector(1.0, 1.0, 0.0);
        let t = view_transform(from, to, up);
        assert_eq!(
            t,
            Matrix4x4::<f32> {
                data: [
                    -0.50709254,
                    0.50709254,
                    0.6761234,
                    -2.366432,
                    0.76771593,
                    0.6060915,
                    0.12121832,
                    -2.828427,
                    -0.35856858,
                    0.59761435,
                    -0.71713716,
                    -2.3841858e-7,
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            }
        );
    }

    #[test]
    fn pixel_sizes() {
        let c = camera(200, 125, PI / 2.0);
        assert_eq!(c.pixel_size, 0.01);
        let c = camera(125, 200, PI / 2.0);
        assert_eq!(c.pixel_size, 0.01);
    }

    #[test]
    fn constructing_rays_through_canvas() {
        let c = camera(201, 101, PI / 2.0);
        let r = c.ray_for_pixel(100, 50);
        assert_eq!(r.origin, point(0.0, 0.0, 0.0));
        assert_eq!(r.direction, vector(0.0, 0.0, -1.0));

        let c = camera(201, 101, PI / 2.0);
        let r = c.ray_for_pixel(0, 0);
        assert_eq!(r.origin, point(0.0, 0.0, 0.0));
        assert_eq!(r.direction, vector(0.6651864, 0.33259323, -0.66851234));

        let mut c = camera(201, 101, PI / 2.0);
        c.transform = rotation_y(PI / 4.0) * translation(0.0, -2.0, 5.0);
        let r = c.ray_for_pixel(100, 50);
        assert_eq!(r.origin, point(0.0, 2.0, -5.0));
        assert_eq!(r.direction, vector(0.70710665, 0.0, -0.7071069));
    }

    #[test]
    fn rendering() {
        let w = default_world();
        let mut c = camera(11, 11, PI / 2.0);
        let from = point(0.0, 0.0, -5.0);
        let to = point(0.0, 0.0, 0.0);
        let up = vector(0.0, 1.0, 0.0);
        c.transform = view_transform(from, to, up);
        let image = render(c, w);
        assert_eq!(
            image.pixel_at(5, 5),
            color(0.38066125, 0.4758265, 0.28549594)
        );
    }

    #[test]
    fn lighting_in_shadow() {
        let m = material();
        let position = point(0.0, 0.0, 0.0);
        let eyev = vector(0.0, 0.0, -1.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0));
        let in_shadow = true;
        let result = lighting(m, light, position, eyev, normalv, in_shadow);
        assert_eq!(result, color(0.1, 0.1, 0.1));
    }

    #[test]
    fn check_shadows() {
        let w = default_world();
        let p = point(0.0, 10.0, 0.0);
        assert_eq!(w.is_shadowed(p), false);
        let p = point(10.0, -10.0, 10.0);
        assert_eq!(w.is_shadowed(p), true);
        let p = point(-20.0, 20.0, -20.0);
        assert_eq!(w.is_shadowed(p), false);
        let p = point(-2.0, 2.0, -2.0);
        assert_eq!(w.is_shadowed(p), false);
    }

    #[test]
    fn intersection_in_shadow() {
        let mut w = world();
        w.light = Some(point_light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0)));
        let s1 = sphere();
        w.objects.push(s1);
        let mut s2 = sphere();
        s2.set_transform(translation(0.0, 0.0, 10.0));
        w.objects.push(s2);
        let r = ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
        let i = intersection(4.0, s2);
        let comps = i.prepare_computations(r, &vec![i]);
        let c = w.shade_hit(comps, 5);
        assert_eq!(c, color(0.1, 0.1, 0.1));
    }

    #[test]
    fn hit_offseting() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut shape = sphere();
        shape.set_transform(translation(0.0, 0.0, 1.0));
        let i = intersection(5.0, shape);
        let comps = i.prepare_computations(r, &vec![i]);
        assert!(comps.over_point.z < -EPSILON / 2.0);
        assert!(comps.point.z > comps.over_point.z);
    }

    #[test]
    fn plane_normals() {
        let p = plane();
        let n1 = p.local_normal_at(point(0.0, 0.0, 0.0));
        let n2 = p.local_normal_at(point(10.0, 0.0, -10.0));
        let n3 = p.local_normal_at(point(-5.0, 0.0, 150.0));
        assert_eq!(n1, vector(0.0, 1.0, 0.0));
        assert_eq!(n2, vector(0.0, 1.0, 0.0));
        assert_eq!(n3, vector(0.0, 1.0, 0.0));
    }

    #[test]
    fn plane_ray_intersect() {
        let p = plane();
        let r = ray(point(0.0, 10.0, 0.0), vector(0.0, 0.0, 1.0));
        let xs = p.local_intersect(r);
        assert!(xs.is_empty());

        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let xs = p.local_intersect(r);
        assert!(xs.is_empty());

        let r = ray(point(0.0, 1.0, 0.0), vector(0.0, -1.0, 0.0));
        let xs = p.local_intersect(r);
        if xs.len() == 1 {
            assert_eq!(xs[0].t, 1.0);
            assert_eq!(xs[0].object, p);
        } else {
            assert!(false);
        }

        let r = ray(point(0.0, -1.0, 0.0), vector(0.0, 1.0, 0.0));
        let xs = p.local_intersect(r);
        if xs.len() == 1 {
            assert_eq!(xs[0].t, 1.0);
            assert_eq!(xs[0].object, p);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn striped_pattern() {
        let pattern = stripe_pattern(white(), black());
        assert_eq!(pattern.a, white());
        assert_eq!(pattern.b, black());

        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 1.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 2.0, 0.0)), white());

        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 1.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 2.0)), white());

        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.9, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(1.0, 0.0, 0.0)), black());
        assert_eq!(pattern.pattern_at(point(-0.1, 0.0, 0.0)), black());
        assert_eq!(pattern.pattern_at(point(-1.0, 0.0, 0.0)), black());
        assert_eq!(pattern.pattern_at(point(-1.1, 0.0, 0.0)), white());
    }

    #[test]
    fn lighting_with_pattern() {
        let mut m = material();
        m.pattern = Some(stripe_pattern(white(), black()));
        m.ambient = 1.0;
        m.diffuse = 0.0;
        m.specular = 0.0;
        let eyev = vector(0.0, 0.0, -1.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, -10.0), white());
        let c1 = lighting(m, light, point(0.9, 0.0, 0.0), eyev, normalv, false);
        let c2 = lighting(m, light, point(1.1, 0.0, 0.0), eyev, normalv, false);
        assert_eq!(c1, white());
        assert_eq!(c2, black());
    }

    #[test]
    fn stripes_with_objects() {
        let mut object = sphere();
        object.transform = scaling(2.0, 2.0, 2.0);
        let pattern = stripe_pattern(white(), black());
        let c = pattern.stripe_at_object(object, point(1.5, 0.0, 0.0));
        assert_eq!(c, white());

        let object = sphere();
        let mut pattern = stripe_pattern(white(), black());
        pattern.transform = scaling(2.0, 2.0, 2.0);
        let c = pattern.stripe_at_object(object, point(1.5, 0.0, 0.0));
        assert_eq!(c, white());

        let mut object = sphere();
        object.transform = scaling(2.0, 2.0, 2.0);
        let mut pattern = stripe_pattern(white(), black());
        pattern.transform = translation(0.5, 0.0, 0.0);
        let c = pattern.stripe_at_object(object, point(2.5, 0.0, 0.0));
        assert_eq!(c, white());
    }

    #[test]
    fn grad_pattern() {
        let pattern = gradient_pattern(white(), black());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(
            pattern.pattern_at(point(0.25, 0.0, 0.0)),
            color(0.75, 0.75, 0.75)
        );
        assert_eq!(
            pattern.pattern_at(point(0.5, 0.0, 0.0)),
            color(0.5, 0.5, 0.5)
        );
        assert_eq!(
            pattern.pattern_at(point(0.75, 0.0, 0.0)),
            color(0.25, 0.25, 0.25)
        );
    }

    #[test]
    fn rng_pattern() {
        let pattern = ring_pattern(white(), black());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(1.0, 0.0, 0.0)), black());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 1.0)), black());
        assert_eq!(pattern.pattern_at(point(0.708, 0.0, 0.708)), black());
    }

    #[test]
    fn check_pattern() {
        let pattern = checkers_pattern(white(), black());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.99, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(1.01, 0.0, 0.0)), black());

        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 0.99, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 1.01, 0.0)), black());

        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.0)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 0.99)), white());
        assert_eq!(pattern.pattern_at(point(0.0, 0.0, 1.01)), black());
    }

    #[test]
    fn precomputing_reflection() {
        let shape = plane();
        let r = ray(
            point(0.0, 1.0, -1.0),
            vector(0.0, -2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
        );
        let i = intersection(2f32.sqrt(), shape);
        let comps = i.prepare_computations(r, &vec![i]);
        assert_eq!(
            comps.reflectv,
            vector(0.0, 2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0)
        )
    }

    #[test]
    fn reflections() {
        let mut w = default_world();
        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        w.objects[1].material.ambient = 1.0;
        let i = intersection(1.0, w.objects[1]);
        let comps = i.prepare_computations(r, &vec![i]);
        let col = w.reflected_color(comps, 5);
        assert_eq!(col, color(0.0, 0.0, 0.0));

        let mut w = default_world();
        let mut shape = plane();
        shape.material.reflective = 0.5;
        shape.transform = translation(0.0, -1.0, 0.0);
        w.objects.push(shape);
        let r = ray(
            point(0.0, 0.0, -3.0),
            vector(0.0, -2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
        );
        let i = intersection(2f32.sqrt(), shape);
        let comps = i.prepare_computations(r, &vec![i]);
        let col = w.reflected_color(comps, 5);
        assert_eq!(col, color(0.190332, 0.237915, 0.1427491));

        let mut w = default_world();
        let mut shape = plane();
        shape.material.reflective = 0.5;
        shape.transform = translation(0.0, -1.0, 0.0);
        w.objects.push(shape);
        let r = ray(
            point(0.0, 0.0, -3.0),
            vector(0.0, -2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
        );
        let i = intersection(2f32.sqrt(), shape);
        let comps = i.prepare_computations(r, &vec![i]);
        let col = w.shade_hit(comps, 5);
        assert_eq!(col, color(0.876757, 0.9243403, 0.829174));
    }

    #[test]
    fn avoid_recursion() {
        let mut w = world();
        w.light = Some(point_light(point(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0)));
        let mut lower = plane();
        lower.material.reflective = 1.0;
        lower.transform = translation(0.0, -1.0, 0.0);
        w.objects.push(lower);
        let mut upper = plane();
        upper.material.reflective = 1.0;
        upper.transform = translation(0.0, 1.0, 0.0);
        w.objects.push(upper);
        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0));
        w.color_at(r, 5);
    }

    #[test]
    fn recursion() {
        let mut w = default_world();
        let mut shape = plane();
        shape.material.reflective = 0.5;
        shape.transform = translation(0.0, -1.0, 0.0);
        w.objects.push(shape);
        let r = ray(
            point(0.0, 0.0, -3.0),
            vector(0.0, -2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
        );
        let i = intersection(2f32.sqrt(), shape);
        let comps = i.prepare_computations(r, &vec![i]);
        let col = w.reflected_color(comps, 0);
        assert_eq!(col, black());
    }

    #[test]
    fn finding_n1_n2() {
        let mut a = glass_sphere();
        a.transform = scaling(2.0, 2.0, 2.0);
        a.material.refractive_index = 1.5;

        let mut b = glass_sphere();
        b.transform = translation(0.0, 0.0, -0.25);
        b.material.refractive_index = 2.0;

        let mut c = glass_sphere();
        c.transform = translation(0.0, 0.0, 0.25);
        c.material.refractive_index = 2.5;

        let r = ray(point(0.0, 0.0, -4.0), vector(0.0, 0.0, 1.0));
        let xs: Vec<Intersection> = vec![
            (2.0, a),
            (2.75, b),
            (3.25, c),
            (4.75, b),
            (5.25, c),
            (6.0, a),
        ]
        .iter()
        .map(|&(t, o)| intersection(t, o))
        .collect();
        let comps = xs[0].prepare_computations(r, &xs);
        assert_eq!(comps.n1, 1.0);
        assert_eq!(comps.n2, 1.5);
        let comps = xs[1].prepare_computations(r, &xs);
        assert_eq!(comps.n1, 1.5);
        assert_eq!(comps.n2, 2.0);
        let comps = xs[2].prepare_computations(r, &xs);
        assert_eq!(comps.n1, 2.0);
        assert_eq!(comps.n2, 2.5);
        let comps = xs[3].prepare_computations(r, &xs);
        assert_eq!(comps.n1, 2.5);
        assert_eq!(comps.n2, 2.5);
        let comps = xs[4].prepare_computations(r, &xs);
        assert_eq!(comps.n1, 2.5);
        assert_eq!(comps.n2, 1.5);
        let comps = xs[5].prepare_computations(r, &xs);
        assert_eq!(comps.n1, 1.5);
        assert_eq!(comps.n2, 1.0);
    }

    #[test]
    fn under_point() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut shape = glass_sphere();
        shape.transform = translation(0.0, 0.0, 1.0);
        let i = intersection(5.0, shape);
        let xs = intersections![i];
        let comps = i.prepare_computations(r, &xs);
        assert!(comps.under_point.z > EPSILON / 2.0);
        assert!(comps.point.z < comps.under_point.z);
    }

    fn test_pattern() -> Pattern {
        Pattern {
            typ: crate::PatType::Test,
            a: white(),
            b: black(),
            transform: identity_matrix(),
        }
    }

    #[test]
    fn refract_color() {
        let mut w = default_world();
        let shape = w.objects[0];
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let xs = intersections![intersection(4.0, shape), intersection(6.0, shape)];
        let comps = xs[0].prepare_computations(r, &xs);
        let c = w.refracted_color(comps, 5);
        assert_eq!(c, color(0.0, 0.0, 0.0));

        w.objects[0].material.transparency = 1.0;
        w.objects[0].material.refractive_index = 1.5;
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let xs = intersections![intersection(4.0, shape), intersection(6.0, shape)];
        let comps = xs[0].prepare_computations(r, &xs);
        let c = w.refracted_color(comps, 0);
        assert_eq!(c, color(0.0, 0.0, 0.0));

        w.objects[0].material.transparency = 1.0;
        w.objects[0].material.refractive_index = 1.5;
        let r = ray(point(0.0, 0.0, 2f32.sqrt() / 2.0), vector(0.0, 1.0, 0.0));
        let xs = intersections![
            intersection(-2f32.sqrt() / 2.0, shape),
            intersection(2f32.sqrt() / 2.0, shape)
        ];
        let comps = xs[1].prepare_computations(r, &xs);
        let c = w.refracted_color(comps, 5);
        assert_eq!(c, color(0.0, 0.0, 0.0));

        let mut w = default_world();
        w.objects[0].material.ambient = 1.0;
        w.objects[0].material.pattern = Some(test_pattern());
        w.objects[1].material.transparency = 1.0;
        w.objects[1].material.refractive_index = 1.5;
        let r = ray(point(0.0, 0.0, 0.1), vector(0.0, 1.0, 0.0));
        let xs = intersections![
            intersection(-0.9899, w.objects[0]),
            intersection(-0.4899, w.objects[1]),
            intersection(0.4899, w.objects[1]),
            intersection(0.9899, w.objects[0])
        ];
        let comps = xs[2].prepare_computations(r, &xs);
        let c = w.refracted_color(comps, 5);
        assert_eq!(c, color(0.0, 0.998874, 0.0472189));
    }

    #[test]
    fn shade_hit_transparent() {
        let mut w = default_world();
        let mut floor = plane();
        floor.transform = translation(0.0, -1.0, 0.0);
        floor.material.transparency = 0.5;
        floor.material.refractive_index = 1.5;
        w.objects.push(floor);
        let mut ball = sphere();
        ball.material.color = color(1.0, 0.0, 0.0);
        ball.material.ambient = 0.5;
        ball.transform = translation(0.0, -3.5, -0.5);
        w.objects.push(ball);
        let r = ray(
            point(0.0, 0.0, -3.0),
            vector(0.0, -2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
        );
        let xs = intersections![intersection(2f32.sqrt(), floor)];
        let comps = xs[0].prepare_computations(r, &xs);
        let c = w.shade_hit(comps, 5);
        assert_eq!(c, color(0.93642, 0.68642, 0.68642));
    }

    #[test]
    fn fresnel() {
        let shape = glass_sphere();
        let r = ray(point(0.0, 0.0, 2f32.sqrt() / 2.0), vector(0.0, 1.0, 0.0));
        let xs = intersections![
            intersection(-2f32.sqrt() / 2.0, shape),
            intersection(2f32.sqrt() / 2.0, shape)
        ];
        let comps = xs[1].prepare_computations(r, &xs);
        let reflectance = comps.schlick();
        assert_eq!(reflectance, 1.0);

        let shape = glass_sphere();
        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0));
        let xs = intersections![intersection(-1.0, shape), intersection(1.0, shape)];
        let comps = xs[1].prepare_computations(r, &xs);
        let reflectance = comps.schlick();
        assert!((reflectance - 0.04).abs() < EPSILON);

        let shape = glass_sphere();
        let r = ray(point(0.0, 0.99, -2.0), vector(0.0, 0.0, 1.0));
        let xs = intersections![intersection(1.8589, shape)];
        let comps = xs[0].prepare_computations(r, &xs);
        let reflectance = comps.schlick();
        assert!((reflectance - 0.48873).abs() < EPSILON);
    }

    #[test]
    fn fresnel_in_shade_hit() {
        let mut w = default_world();
        let r = ray(
            point(0.0, 0.0, -3.0),
            vector(0.0, -2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0),
        );
        let mut floor = plane();
        floor.transform = translation(0.0, -1.0, 0.0);
        floor.material.reflective = 0.5;
        floor.material.transparency = 0.5;
        floor.material.refractive_index = 1.5;
        w.objects.push(floor);
        let mut ball = sphere();
        ball.material.color = color(1.0, 0.0, 0.0);
        ball.material.ambient = 0.5;
        ball.transform = translation(0.0, -3.5, -0.5);
        w.objects.push(ball);
        let xs = intersections!(intersection(2f32.sqrt(), floor));
        let comps = xs[0].prepare_computations(r, &xs);
        let c = w.shade_hit(comps, 5);
        assert_eq!(c, color(0.93391, 0.69643, 0.69243));
    }

    #[test]
    fn a_cube() {
        let examples = vec![
            (point(5.0, 0.5, 0.0), vector(-1.0, 0.0, 0.0), 4.0, 6.0),
            (point(-5.0, 0.5, 0.0), vector(1.0, 0.0, 0.0), 4.0, 6.0),
            (point(0.5, 5.0, 0.0), vector(0.0, -1.0, 0.0), 4.0, 6.0),
            (point(0.5, -5.0, 0.0), vector(0.0, 1.0, 0.0), 4.0, 6.0),
            (point(0.5, 0.0, 5.0), vector(0.0, 0.0, -1.0), 4.0, 6.0),
            (point(0.5, 0.0, -5.0), vector(0.0, 0.0, 1.0), 4.0, 6.0),
            (point(0.0, 0.5, 0.0), vector(0.0, 0.0, 1.0), -1.0, 1.0),
        ];
        let c = cube();
        for &(or, dir, t1, t2) in examples.iter() {
            let r = ray(or, dir);
            let xs = c.local_intersect(r);
            if xs.len() == 2 {
                assert_eq!(xs[0].t, t1);
                assert_eq!(xs[1].t, t2);
            } else {
                assert!(false);
            }
        }
    }

    #[test]
    fn missed_the_cube() {
        let examples = vec![
            (point(-2.0, 0.0, 0.0), vector(0.2673, 0.5345, 0.8018)),
            (point(0.0, -2.0, 0.0), vector(0.8018, 0.2673, 0.5345)),
            (point(0.0, 0.0, -2.0), vector(0.5345, 0.8018, 0.2673)),
            (point(2.0, 0.0, 2.0), vector(0.0, 0.0, -1.0)),
            (point(0.0, 2.0, 2.0), vector(0.0, -1.0, 0.0)),
            (point(2.0, 2.0, 0.0), vector(-1.0, 0.0, 0.0)),
        ];
        let c = cube();
        for &(or, dir) in examples.iter() {
            let r = ray(or, dir);
            let xs = c.local_intersect(r);
            assert!(xs.len() == 0);
        }
    }

    #[test]
    fn normals_on_a_cube() {
        let examples = vec![
            (point(1.0, 0.5, -0.8), vector(1.0, 0.0, 0.0)),
            (point(-1.0, -0.2, 0.9), vector(-1.0, 0.0, 0.0)),
            (point(-0.4, 1.0, -0.1), vector(0.0, 1.0, 0.0)),
            (point(0.3, -1.0, -0.7), vector(0.0, -1.0, 0.0)),
            (point(-0.6, 0.3, 1.0), vector(0.0, 0.0, 1.0)),
            (point(0.4, 0.4, -1.0), vector(0.0, 0.0, -1.0)),
            (point(1.0, 1.0, 1.0), vector(1.0, 0.0, 0.0)),
            (point(-1.0, -1.0, -1.0), vector(-1.0, 0.0, 0.0)),
        ];
        let c = cube();
        for &(p, norm) in examples.iter() {
            let n = c.local_normal_at(p);
            assert_eq!(n, norm);
        }
    }

    #[test]
    fn missing_the_cylinder() {
        let examples = vec![
            (point(1.0, 0.0, 0.0), vector(0.0, 1.0, 0.0)),
            (point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0)),
            (point(0.0, 0.0, -5.0), vector(1.0, 1.0, 1.1)),
        ];
        let c = cylinder();
        for &(or, dir) in examples.iter() {
            let d = dir.normalize();
            let r = ray(or, d);
            let xs = c.local_intersect(r);
            assert!(xs.len() == 0);
        }
    }

    #[test]
    fn hitting_the_cylinder() {
        let examples = vec![
            (point(1.0, 0.0, -5.0), vector(0.0, 0.0, 1.0), 5.0, 5.0),
            (point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0), 4.0, 6.0),
            (
                point(0.5, 0.0, -5.0),
                vector(0.1, 1.0, 1.0),
                6.808006,
                7.0886984,
            ),
        ];
        let cyl = cylinder();
        for &(or, dir, t1, t2) in examples.iter() {
            let d = dir.normalize();
            let r = ray(or, d);
            let xs = cyl.local_intersect(r);
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, t1);
            assert_eq!(xs[1].t, t2);
        }
    }

    #[test]
    fn normals_on_a_cylinder() {
        let examples = vec![
            (point(1.0, 0.0, 0.0), vector(1.0, 0.0, 0.0)),
            (point(0.0, 5.0, -1.0), vector(0.0, 0.0, -1.0)),
            (point(0.0, -2.0, 1.0), vector(0.0, 0.0, 1.0)),
            (point(-1.0, 1.0, 0.0), vector(-1.0, 0.0, 0.0)),
        ];
        let c = cylinder();
        for &(p, norm) in examples.iter() {
            let n = c.local_normal_at(p);
            assert_eq!(n, norm);
        }
    }

    #[test]
    fn intersect_truncated_cylinder() {
        let examples = vec![
            (point(0.0, 1.5, 0.0), vector(0.1, 1.0, 0.0), 0),
            (point(0.0, 3.0, -5.0), vector(0.0, 0.0, 1.0), 0),
            (point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0), 0),
            (point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0), 0),
            (point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0), 0),
            (point(0.0, 1.5, -2.0), vector(0.0, 0.0, 1.0), 2),
        ];
        let mut cyl = cylinder();
        cyl.typ = ShapeType::Cylinder(1.0, 2.0, false);
        for &(p, dir, count) in examples.iter() {
            let d = dir.normalize();
            let r = ray(p, d);
            let xs = cyl.local_intersect(r);
            assert_eq!(xs.len(), count);
        }
    }

    #[test]
    fn capped_cylinder() {
        let examples = vec![
            (point(0.0, 3.0, 0.0), vector(0.0, -1.0, 0.0), 2),
            (point(0.0, 3.0, -2.0), vector(0.0, -1.0, 2.0), 2),
            (point(0.0, 4.0, -2.0), vector(0.0, -1.0, 1.0), 2),
            (point(0.0, 0.0, -2.0), vector(0.0, 1.0, 2.0), 2),
            (point(0.0, -1.0, -2.0), vector(0.0, 1.0, 1.0), 2),
        ];
        let mut cyl = cylinder();
        cyl.typ = ShapeType::Cylinder(1.0, 2.0, true);
        for &(or, dir, count) in examples.iter() {
            let d = dir.normalize();
            let r = ray(or, d);
            let xs = cyl.local_intersect(r);
            assert_eq!(xs.len(), count);
        }
    }

    #[test]
    fn normal_vector_at_end() {
        let examples = vec![
            (point(0.0, 1.0, 0.0), vector(0.0, -1.0, 0.0)),
            (point(0.5, 1.0, 0.0), vector(0.0, -1.0, 0.0)),
            (point(0.0, 1.0, 0.5), vector(0.0, -1.0, 0.0)),
            (point(0.0, 2.0, 0.0), vector(0.0, 1.0, 0.0)),
            (point(0.5, 2.0, 0.0), vector(0.0, 1.0, 0.0)),
            (point(0.0, 2.0, 0.5), vector(0.0, 1.0, 0.0)),
        ];
        let mut cyl = cylinder();
        cyl.typ = ShapeType::Cylinder(1.0, 2.0, true);
        for &(p, norm) in examples.iter() {
            let n = cyl.local_normal_at(p);
            assert_eq!(n, norm);
        }
    }

    #[test]
    fn intersecting_a_cone() {
        let examples = vec![
            (point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0), 5.0, 5.0),
            (
                point(0.0, 0.0, -5.0),
                vector(1.0, 1.0, 1.0),
                8.66025,
                8.66025,
            ),
            (
                point(1.0, 1.0, -5.0),
                vector(-0.5, -1.0, -1.0),
                4.55006,
                49.44994,
            ),
        ];
        let shape = cone();
        for &(or, dir, t0, t1) in examples.iter() {
            let d = dir.normalize();
            let r = ray(or, d);
            let xs = shape.local_intersect(r);
            assert_eq!(xs.len(), 2);
            assert!((xs[0].t - t0).abs() < EPSILON);
            assert!((xs[1].t - t1).abs() < EPSILON);
        }
        let dir = vector(0.0, 1.0, 1.0).normalize();
        let r = ray(point(0.0, 0.0, -1.0), dir);
        let xs = shape.local_intersect(r);
        assert_eq!(xs.len(), 1);
        assert!((xs[0].t - 0.35355).abs() < EPSILON);
    }
}
