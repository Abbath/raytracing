use std::ops;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Tuple {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

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
    fn normalize(self) -> Tuple {
        let m = self.magnitude();
        Tuple {
            x: self.x / m,
            y: self.y / m,
            z: self.z / m,
            w: self.w / m,
        }
    }
    fn dot(self, other: Tuple) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
    fn cross(self, other: Tuple) -> Tuple {
        vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
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

fn point(x: f32, y: f32, z: f32) -> Tuple {
    Tuple { x, y, z, w: 1.0f32 }
}

fn vector(x: f32, y: f32, z: f32) -> Tuple {
    Tuple { x, y, z, w: 0.0f32 }
}

fn color(x: f32, y: f32, z: f32) -> Tuple {
    Tuple { x, y, z, w: 0.0f32 }
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
    fn at(&self, r: usize, c: usize) -> T {
        let idx = r * 2 + c;
        self.data[idx]
    }
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
    fn pixel_at(self, x: usize, y: usize) -> Tuple {
        let idx = x * self.h + y;
        self.m[idx]
    }
    fn write_pixel(&mut self, x: usize, y: usize, c: Tuple) {
        let idx = x * self.h + y;
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
    origin: Tuple,
    direction: Tuple,
}

fn ray(origin: Tuple, direction: Tuple) -> Ray {
    Ray { origin, direction }
}

impl Ray {
    fn position(self, t: f32) -> Tuple {
        self.origin + self.direction * t
    }
    fn transform(self, t: Matrix4x4<f32>) -> Ray {
        let o = t * self.origin;
        let d = t * self.direction;
        Ray {
            origin: o,
            direction: d,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Sphere {
    origin: Tuple,
    radius: f32,
    transform: Matrix4x4<f32>,
}

fn sphere() -> Sphere {
    Sphere {
        origin: point(0.0, 0.0, 0.0),
        radius: 1.0,
        transform: identity_matrix(),
    }
}

impl Sphere {
    fn intersect(self, ray: Ray) -> Vec<Intersection> {
        let r = ray.transform(self.transform.inverse());
        let sphere_to_ray = r.origin - self.origin;
        let a = r.direction.dot(r.direction);
        let b = 2.0 * r.direction.dot(sphere_to_ray);
        let c = sphere_to_ray.dot(sphere_to_ray) - 1.0;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return intersections![];
        }
        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
        intersections![
            intersection(t1, InterObject::S(self)),
            intersection(t2, InterObject::S(self))
        ]
    }
    fn set_transform(&mut self, t: Matrix4x4<f32>) {
        self.transform = t;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum InterObject {
    S(Sphere),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Intersection {
    t: f32,
    object: InterObject,
}

fn intersection(t: f32, o: InterObject) -> Intersection {
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

fn main() {
    let ray_origin = point(0.0, 0.0, -5.0);
    let wall_z = 10.0;
    let wall_size = 7.0;
    let canvas_pixels = 100;
    let pixel_size = wall_size / canvas_pixels as f32;
    let half = wall_size / 2.0;
    let mut canvas = canvas(canvas_pixels, canvas_pixels);
    let color = color(1.0, 0.0, 0.0);
    let shape = sphere();
    for y in 0..canvas_pixels {
        let world_y = half - pixel_size * y as f32;
        for x in 0..canvas_pixels {
            let world_x = -half + pixel_size * x as f32;
            let position = point(world_x, world_y, wall_z);
            let r = ray(ray_origin, (position - ray_origin).normalize());
            let xs = shape.intersect(r);
            if hit(xs).is_some() {
                canvas.write_pixel(x, y, color);
            }
        }
    }
    println!("{}", canvas.to_ppm());
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::{
        canvas, color, hit, identity_matrix, intersection, intersections, point, ray, rotation_x,
        rotation_y, rotation_z, scaling, shearing, sphere, translation, tuple, vector, Matrix2x2,
        Matrix3x3, Matrix4x4,
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
        assert!((1.0 - vector(1.0, 2.0, 3.0).normalize().magnitude()).abs() < f32::EPSILON);
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
            vector(0.9, 0.2, 0.040000003),
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
        assert_eq!(point(0.0, -4.371139e-8, 1.0), full_quarter * p);
        let inv = half_quarter.inverse();
        assert_eq!(point(0.0, 0.7071068, -0.7071068), inv * p);
        let p = point(0.0, 0.0, 1.0);
        let half_quarter = rotation_y(PI / 4.0);
        let full_quarter = rotation_y(PI / 2.0);
        assert_eq!(
            point(2f32.sqrt() / 2.0, 0.0, 2f32.sqrt() / 2.0),
            half_quarter.clone() * p
        );
        assert_eq!(point(1.0, 0.0, -4.371139e-8), full_quarter * p);
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_z(PI / 4.0);
        let full_quarter = rotation_z(PI / 2.0);
        assert_eq!(
            point(-2f32.sqrt() / 2.0, 2f32.sqrt() / 2.0, 0.0),
            half_quarter.clone() * p
        );
        assert_eq!(point(-1.0, -4.371139e-8, 0.0), full_quarter * p);
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
        assert_eq!(p2, point(1.0, -1.0, -4.371139e-8));
        let p3 = b * p2;
        assert_eq!(p3, point(5.0, -5.0, -2.1855695e-7));
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
            assert_eq!(xs[0].object, crate::InterObject::S(s));
            assert_eq!(xs[1].object, crate::InterObject::S(s));
        } else {
            assert!(false);
        }
    }

    #[test]
    fn aggregate_intersections() {
        let s = sphere();
        let i = intersection(3.5, crate::InterObject::S(s));
        assert_eq!(i.t, 3.5);
        assert_eq!(i.object, crate::InterObject::S(s));

        let s = sphere();
        let i1 = intersection(1.0, crate::InterObject::S(s));
        let i2 = intersection(2.0, crate::InterObject::S(s));
        let xs = intersections![i1, i2];
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[1].t, 2.0);
    }

    #[test]
    fn hits() {
        let s = sphere();
        let i1 = intersection(1.0, crate::InterObject::S(s));
        let i2 = intersection(2.0, crate::InterObject::S(s));
        let xs = intersections![i1, i2];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i1);
        } else {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(-1.0, crate::InterObject::S(s));
        let i2 = intersection(1.0, crate::InterObject::S(s));
        let xs = intersections![i1, i2];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i2);
        } else {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(-2.0, crate::InterObject::S(s));
        let i2 = intersection(-1.0, crate::InterObject::S(s));
        let xs = intersections![i1, i2];
        if let Some(_) = hit(xs) {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(5.0, crate::InterObject::S(s));
        let i2 = intersection(7.0, crate::InterObject::S(s));
        let i3 = intersection(-3.0, crate::InterObject::S(s));
        let i4 = intersection(2.0, crate::InterObject::S(s));
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
}
