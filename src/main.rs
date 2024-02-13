use std::f32::EPSILON;
use std::iter::Once;
use std::ops;

#[derive(Debug, Clone, Copy)]
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
    fn reflect(self, normal: Tuple) -> Tuple {
        self - normal * 2.0 * self.dot(normal)
    }
}

impl PartialEq<Tuple> for Tuple {
    fn eq(&self, other: &Tuple) -> bool {
        let f = |a: f32, b: f32| (a - b).abs() < 0.00001;
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
        let idx = y * self.w + x;
        self.m[idx]
    }
    fn write_pixel(&mut self, x: usize, y: usize, c: Tuple) {
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
    material: Material,
}

fn sphere() -> Sphere {
    Sphere {
        origin: point(0.0, 0.0, 0.0),
        radius: 1.0,
        transform: identity_matrix(),
        material: material(),
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
            intersection(t1, Object::S(self)),
            intersection(t2, Object::S(self))
        ]
    }
    fn set_transform(&mut self, t: Matrix4x4<f32>) {
        self.transform = t;
    }
    fn normal_at(self, wp: Tuple) -> Tuple {
        let op = self.transform.inverse() * wp;
        let on = op - self.origin;
        let mut wn = self.transform.inverse().transposed() * on;
        wn.w = 0.0;
        wn.normalize()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Object {
    S(Sphere),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Intersection {
    t: f32,
    object: Object,
}

fn intersection(t: f32, o: Object) -> Intersection {
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
    position: Tuple,
    intensity: Tuple,
}

fn point_light(position: Tuple, intensity: Tuple) -> Light {
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
}

fn material() -> Material {
    Material {
        color: color(1.0, 1.0, 1.0),
        ambient: 0.1,
        diffuse: 0.9,
        specular: 0.9,
        shininess: 200.0,
    }
}

fn black() -> Tuple {
    color(0.0, 0.0, 0.0)
}

fn lighting(mat: Material, light: Light, p: Tuple, eyev: Tuple, normalv: Tuple, in_shadow: bool) -> Tuple {
    let eff_color = mat.color * light.intensity;
    let lightv = (light.position - p).normalize();
    let ambient = eff_color * mat.ambient;
    let light_dot_normal = lightv.dot(normalv);
    let diffuse: Tuple;
    let specular: Tuple;
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
    objects: Vec<Object>,
}

fn world() -> World {
    World {
        light: None,
        objects: vec![],
    }
}

fn default_world() -> World {
    let mut w = world();
    w.light = Some(point_light(
        point(-10.0, 10.0, -10.0),
        color(1.0, 1.0, 1.0),
    ));
    let mut s1 = sphere();
    s1.material.color = color(0.8, 1.0, 0.6);
    s1.material.diffuse = 0.7;
    s1.material.specular = 0.2;
    w.objects.push(Object::S(s1));
    let mut s2 = sphere();
    s2.set_transform(scaling(0.5, 0.5, 0.5));
    w.objects.push(Object::S(s2));
    w
}

impl World {
    fn intersect(&self, r: Ray) -> Vec<Intersection> {
        let mut v: Vec<Intersection> = self
            .objects
            .iter()
            .flat_map(|o| match o {
                Object::S(s) => s.intersect(r),
            })
            .collect();
        v.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        v
    }
    fn shade_hit(&self, comps: Computations) -> Tuple {
        lighting(
            match comps.object {
                Object::S(s) => s.material,
            },
            self.light.unwrap(),
            comps.over_point,
            comps.eyev,
            comps.normalv,
            self.is_shadowed(comps.over_point)
        )
    }
    fn color_at(&self, r: Ray) -> Tuple {
        let xs = self.intersect(r);
        if let Some(x) = hit(xs) {
            let comps = x.prepare_computations(r);
            self.shade_hit(comps)
        } else {
            black()
        }
    }
    fn is_shadowed(&self, p: Tuple) -> bool {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Computations {
    t: f32,
    object: Object,
    point: Tuple,
    eyev: Tuple,
    normalv: Tuple,
    inside: bool,
    over_point: Tuple
}

impl Intersection {
    fn prepare_computations(&self, r: Ray) -> Computations {
        let p = r.position(self.t);
        let mut comps = Computations {
            t: self.t,
            object: self.object,
            point: p,
            eyev: -r.direction,
            normalv: match self.object {
                Object::S(s) => s.normal_at(p),
            },
            inside: false,
            over_point: point(0.0, 0.0, 0.0)
        };
        if comps.normalv.dot(comps.eyev) < 0.0 {
            comps.inside = true;
            comps.normalv = -comps.normalv
        }
        comps.over_point = comps.point + comps.normalv * 0.00001;
        comps
    }
}

fn view_transform(from: Tuple, to: Tuple, up: Tuple) -> Matrix4x4<f32> {
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
            let color = w.color_at(r);
            image.write_pixel(x, y, color);
        }
    }
    image
}

fn main1() {
    let ray_origin = point(0.0, 0.0, -5.0);
    let wall_z = 10.0;
    let wall_size = 7.0;
    let canvas_pixels = 256;
    let pixel_size = wall_size / canvas_pixels as f32;
    let half = wall_size / 2.0;
    let mut canvas = canvas(canvas_pixels, canvas_pixels);
    let mut shape = sphere();
    shape.material = material();
    shape.material.color = color(1.0, 0.2, 1.0);
    let light_pos = point(-10.0, 10.0, -10.0);
    let light_col = color(1.0, 1.0, 1.0);
    let light = point_light(light_pos, light_col);
    for y in 0..canvas_pixels {
        let world_y = half - pixel_size * y as f32;
        for x in 0..canvas_pixels {
            let world_x = -half + pixel_size * x as f32;
            let position = point(world_x, world_y, wall_z);
            let r = ray(ray_origin, (position - ray_origin).normalize());
            let xs = shape.intersect(r);
            if let Some(t) = hit(xs) {
                let p = r.position(t.t);
                let Object::S(s) = t.object;
                let norm = s.normal_at(p);
                let eye = -r.direction;
                let colr = lighting(s.material, light, p, eye, norm, false);
                canvas.write_pixel(x, y, colr);
            }
        }
    }
    println!("{}", canvas.to_ppm());
}

fn main() {
    use std::f32::consts::PI;
    let mut floor = sphere();
    floor.transform = scaling(10.0, 0.01, 10.0);
    floor.material = material();
    floor.material.color = color(1.0, 0.9, 0.9);
    floor.material.specular = 0.0;

    let mut left_wall = sphere();
    left_wall.transform = translation(0.0, 0.0, 5.0)
        * rotation_y(-PI / 4.0)
        * rotation_x(PI / 2.0)
        * scaling(10.0, 0.01, 10.0);
    left_wall.material = floor.material;

    let mut right_wall = sphere();
    right_wall.transform = translation(0.0, 0.0, 5.0)
        * rotation_y(PI / 4.0)
        * rotation_x(PI / 2.0)
        * scaling(10.0, 0.01, 10.0);
    right_wall.material = floor.material;

    let mut middle = sphere();
    middle.transform = translation(-0.5, 1.0, 0.5);
    middle.material = material();
    middle.material.color = color(0.1, 1.0, 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;

    let mut right = sphere();
    right.transform = translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.5);
    right.material = material();
    right.material.color = color(0.5, 1.0, 0.1);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;

    let mut left = sphere();
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33);
    left.material = material();
    left.material.color = color(1.0, 0.8, 0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;

    let mut w = world();
    w.light = Some(point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0)));
    w.objects = vec![floor, left_wall, right_wall, middle, right, left]
        .iter()
        .map(|&s| Object::S(s))
        .collect();
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
        camera, canvas, color, default_world, hit, identity_matrix, intersection, intersections, lighting, material, point, point_light, ray, render, rotation_x, rotation_y, rotation_z, scaling, shearing, sphere, translation, tuple, vector, view_transform, world, Matrix2x2, Matrix3x3, Matrix4x4, Object
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
        assert!((1.0 - vector(1.0, 2.0, 3.0).normalize().magnitude()).abs() < 0.00001);
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
            assert_eq!(xs[0].object, crate::Object::S(s));
            assert_eq!(xs[1].object, crate::Object::S(s));
        } else {
            assert!(false);
        }
    }

    #[test]
    fn aggregate_intersections() {
        let s = sphere();
        let i = intersection(3.5, crate::Object::S(s));
        assert_eq!(i.t, 3.5);
        assert_eq!(i.object, crate::Object::S(s));

        let s = sphere();
        let i1 = intersection(1.0, crate::Object::S(s));
        let i2 = intersection(2.0, crate::Object::S(s));
        let xs = intersections![i1, i2];
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[1].t, 2.0);
    }

    #[test]
    fn hits() {
        let s = sphere();
        let i1 = intersection(1.0, crate::Object::S(s));
        let i2 = intersection(2.0, crate::Object::S(s));
        let xs = intersections![i1, i2];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i1);
        } else {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(-1.0, crate::Object::S(s));
        let i2 = intersection(1.0, crate::Object::S(s));
        let xs = intersections![i1, i2];
        if let Some(i) = hit(xs) {
            assert_eq!(i, i2);
        } else {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(-2.0, crate::Object::S(s));
        let i2 = intersection(-1.0, crate::Object::S(s));
        let xs = intersections![i1, i2];
        if let Some(_) = hit(xs) {
            assert!(false);
        }

        let s = sphere();
        let i1 = intersection(5.0, crate::Object::S(s));
        let i2 = intersection(7.0, crate::Object::S(s));
        let i3 = intersection(-3.0, crate::Object::S(s));
        let i4 = intersection(2.0, crate::Object::S(s));
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
        let i = intersection(4.0, crate::Object::S(shape));
        let comps = i.prepare_computations(r);
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
        let i = intersection(4.0, crate::Object::S(shape));
        let comps = i.prepare_computations(r);
        assert_eq!(comps.inside, false);

        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let shape = sphere();
        let i = intersection(1.0, crate::Object::S(shape));
        let comps = i.prepare_computations(r);
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
        let comps = i.prepare_computations(r);
        let c = w.shade_hit(comps);
        assert_eq!(c, color(0.38066125, 0.4758265, 0.28549594));

        let mut w = default_world();
        w.light = Some(point_light(point(0.0, 0.25, 0.0), color(1.0, 1.0, 1.0)));
        let r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let shape = w.objects[1];
        let i = intersection(0.5, shape);
        let comps = i.prepare_computations(r);
        let c = w.shade_hit(comps);
        assert_eq!(c, color(0.9049845, 0.9049845, 0.9049845));
    }

    #[test]
    fn color_at() {
        let w = default_world();
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));
        let c = w.color_at(r);
        assert_eq!(c, color(0.0, 0.0, 0.0));

        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let c = w.color_at(r);
        assert_eq!(c, color(0.38066125, 0.4758265, 0.28549594));

        let mut w = default_world();
        let crate::Object::S(mut outer) = w.objects[0];
        let crate::Object::S(mut inner) = w.objects[1];
        outer.material.ambient = 1.0;
        inner.material.ambient = 1.0;
        w.objects[0] = crate::Object::S(outer);
        w.objects[1] = crate::Object::S(inner);
        let r = ray(point(0.0, 0.0, 0.75), vector(0.0, 0.0, -1.0));
        let c = w.color_at(r);
        assert_eq!(c, inner.material.color);
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
        w.objects.push(Object::S(s1));
        let mut s2 = sphere();
        s2.set_transform(translation(0.0, 0.0, 10.0));
        w.objects.push(Object::S(s2));
        let r = ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
        let i = intersection(4.0, Object::S(s2));
        let comps = i.prepare_computations(r);
        let c= w.shade_hit(comps);
        assert_eq!(c, color(0.1, 0.1, 0.1));
    }

    #[test]
    fn hit_offseting() {
        let r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut shape = sphere();
        shape.set_transform(translation(0.0, 0.0, 1.0));
        let i = intersection(5.0, Object::S(shape));
        let comps = i.prepare_computations(r);
        assert!(comps.over_point.z < -0.00001 / 2.0);
        assert!(comps.point.z > comps.over_point.z);
    }
}
