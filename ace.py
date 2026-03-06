import vtk
from vmtk import vmtkscripts
import numpy as np
import os
import glob
from pathlib import Path


# ===================== UTILIDADES DE GEOMETRÍA =====================

def project_point_to_surface(point, surface):
    """Proyecta un punto sobre la superficie más cercana usando el método correcto"""
    try:
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(surface)
        cell_locator.BuildLocator()

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        cell_locator.FindClosestPoint(point, closest_point, cell_id, sub_id, dist2)
        return closest_point
    except Exception:
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(surface)
        locator.BuildLocator()
        closest_point_id = locator.FindClosestPoint(point)
        projected_point = surface.GetPoint(closest_point_id)
        return projected_point


def get_surface_normals(surface, point):
    """Obtiene el vector normal de la superficie en un punto dado usando interpolación"""
    try:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.Update()

        surface_with_normals = normals.GetOutput()

        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(surface_with_normals)
        cell_locator.BuildLocator()

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)
        cell_locator.FindClosestPoint(point, closest_point, cell_id, sub_id, dist2)

        cell_normals = surface_with_normals.GetCellData().GetNormals()
        if cell_normals and cell_id.get() >= 0:
            normal = cell_normals.GetTuple(cell_id.get())
            return normal

        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(surface_with_normals)
        point_locator.BuildLocator()

        closest_point_id = point_locator.FindClosestPoint(point)
        point_normals = surface_with_normals.GetPointData().GetNormals()
        if point_normals:
            normal = point_normals.GetTuple(closest_point_id)
            return normal

        return (0.0, 0.0, -1.0)
    except Exception:
        return (0.0, 0.0, -1.0)


# ===================== VERIFICACIÓN DE RADIO Y DIRECCIÓN =====================

def verify_point_has_surrounding_mesh(point, surface, radius_factor=0.8):
    """Verifica si un punto tiene malla circundante en su radio esperado"""
    try:
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(surface)
        cell_locator.BuildLocator()
        
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(surface)
        
        surface_distance = abs(implicit.EvaluateFunction(point))
        reference_radius = surface_distance * radius_factor
        
        if reference_radius < 0.1:
            reference_radius = 0.5
        
        num_samples = 12
        directions = []
        
        for i in range(num_samples):
            theta = 2 * np.pi * i / num_samples
            for j in range(3):
                phi = np.pi * (j + 1) / 4
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                directions.append([x, y, z])
        
        intersections_found = 0
        total_directions = len(directions)
        
        for direction in directions:
            test_point = [
                point[0] + direction[0] * reference_radius,
                point[1] + direction[1] * reference_radius,
                point[2] + direction[2] * reference_radius
            ]
            
            ray_start = point
            ray_end = test_point
            
            tolerance = 0.001
            t = vtk.mutable(0.0)
            x = [0.0, 0.0, 0.0]
            pcoords = [0.0, 0.0, 0.0]
            subId = vtk.mutable(0)
            cellId = vtk.mutable(0)
            
            intersected = cell_locator.IntersectWithLine(
                ray_start, ray_end, tolerance, t, x, pcoords, subId, cellId
            )
            
            if intersected:
                intersections_found += 1
        
        coverage_ratio = intersections_found / total_directions
        is_well_surrounded = coverage_ratio >= 0.6
        
        return is_well_surrounded, coverage_ratio, reference_radius
        
    except Exception:
        return False, 0.0, 0.0


def get_centerline_direction_from_start(centerline_points, start_index=0):
    """Calcula el vector direccional de una centerline desde el punto de inicio"""
    if len(centerline_points) < 2:
        return [0.0, 0.0, 0.0]
    
    max_points = min(5, len(centerline_points) - 1)
    direction_vectors = []
    
    start_point = np.array(centerline_points[start_index])
    
    for i in range(start_index + 1, start_index + max_points + 1):
        if i < len(centerline_points):
            next_point = np.array(centerline_points[i])
            direction = next_point - start_point
            if np.linalg.norm(direction) > 0:
                direction_vectors.append(direction / np.linalg.norm(direction))
    
    if not direction_vectors:
        return [0.0, 0.0, 0.0]
    
    avg_direction = np.mean(direction_vectors, axis=0)
    if np.linalg.norm(avg_direction) > 0:
        avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
    return avg_direction.tolist()


def trim_centerline_from_start(centerline_points, start_point, surface, max_extension_ratio=0.00001):
    """Recorta una centerline desde el inicio, eliminando puntos que se extienden fuera del modelo"""
    if len(centerline_points) < 2:
        return centerline_points
    
    distances_to_start = []
    for i, point in enumerate(centerline_points):
        dist = np.linalg.norm(np.array(point) - np.array(start_point))
        distances_to_start.append((i, dist, point))
    
    distances_to_start.sort(key=lambda x: x[1])
    closest_index = distances_to_start[0][0]
    
    total_length = 0.0
    for i in range(1, len(centerline_points)):
        segment_length = np.linalg.norm(
            np.array(centerline_points[i]) - np.array(centerline_points[i-1])
        )
        total_length += segment_length
    
    max_allowed_extension = total_length * max_extension_ratio
    
    main_direction = get_centerline_direction_from_start(centerline_points, closest_index)
    
    valid_start_index = closest_index
    valid_end_index = len(centerline_points) - 1
    
    for i in range(closest_index - 1, -1, -1):
        point = centerline_points[i]
        distance_from_start = np.linalg.norm(np.array(point) - np.array(start_point))
        
        if distance_from_start > max_allowed_extension:
            break
        
        is_surrounded, coverage, radius = verify_point_has_surrounding_mesh(point, surface)
        
        if not is_surrounded:
            break
        
        if i < closest_index:
            point_vector = np.array(point) - np.array(start_point)
            if np.linalg.norm(point_vector) > 0:
                point_vector = point_vector / np.linalg.norm(point_vector)
                dot_product = np.dot(point_vector, main_direction)
                
                if dot_product > 0.5:
                    break
        
        valid_start_index = i
    
    for i in range(len(centerline_points) - 1, closest_index, -1):
        point = centerline_points[i]
        is_surrounded, coverage, radius = verify_point_has_surrounding_mesh(point, surface)
        
        if not is_surrounded:
            valid_end_index = i - 1
            break
    
    if valid_start_index != 0 or valid_end_index != len(centerline_points) - 1:
        trimmed_points = centerline_points[valid_start_index:valid_end_index + 1]
        return trimmed_points
    else:
        return centerline_points


# ===================== MÉTODOS PARA ENCONTRAR CENTRO DEL LUMEN =====================

def find_center_by_grid_sampling(surface, opening_center, opening_radius, grid_density=10):
    """Encuentra el centro del lumen usando muestreo en grilla 3D"""
    try:
        implicit_surface = vtk.vtkImplicitPolyDataDistance()
        implicit_surface.SetInput(surface)
        
        search_radius = opening_radius * 2.0
        step = search_radius / grid_density
        
        best_point = None
        max_distance = -1.0
        
        for i in range(-grid_density, grid_density + 1):
            for j in range(-grid_density, grid_density + 1):
                for k in range(-grid_density, grid_density + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                        
                    test_point = [
                        opening_center[0] + i * step,
                        opening_center[1] + j * step,
                        opening_center[2] + k * step
                    ]
                    
                    distance = implicit_surface.EvaluateFunction(test_point)
                    
                    if distance < 0 and abs(distance) > max_distance:
                        max_distance = abs(distance)
                        best_point = test_point.copy()
        
        return best_point
            
    except Exception:
        return None


def find_center_by_random_sampling(surface, opening_center, opening_radius, num_samples=100):
    """Encuentra el centro del lumen usando muestreo aleatorio"""
    try:
        implicit_surface = vtk.vtkImplicitPolyDataDistance()
        implicit_surface.SetInput(surface)
        
        best_point = None
        max_distance = -1.0
        search_radius = opening_radius * 2.0
        
        for _ in range(num_samples):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, search_radius)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            test_point = [
                opening_center[0] + x,
                opening_center[1] + y,
                opening_center[2] + z
            ]
            
            distance = implicit_surface.EvaluateFunction(test_point)
            
            if distance < 0 and abs(distance) > max_distance:
                max_distance = abs(distance)
                best_point = test_point.copy()
        
        return best_point
            
    except Exception:
        return None


def find_center_by_surface_projection(surface, opening_center, opening_radius):
    """Encuentra el centro del lumen proyectando desde el borde de la abertura hacia adentro"""
    try:
        normal = get_surface_normals(surface, opening_center)
        normal = np.array(normal)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        
        implicit_surface = vtk.vtkImplicitPolyDataDistance()
        implicit_surface.SetInput(surface)
        
        max_depth = opening_radius * 3.0
        num_steps = 20
        
        best_point = None
        max_distance = -1.0
        
        for i in range(1, num_steps + 1):
            depth = (max_depth / num_steps) * i
            test_point = np.array(opening_center) - normal * depth
            
            distance = implicit_surface.EvaluateFunction(test_point)
            
            if distance < 0 and abs(distance) > max_distance:
                max_distance = abs(distance)
                best_point = test_point.copy()
        
        if best_point is not None:
            return best_point.tolist()
        else:
            return None
            
    except Exception:
        return None


def find_center_by_max_distance_with_reference(surface, opening_center, reference_distance, num_samples=25):
    try:
        implicit_surface = vtk.vtkImplicitPolyDataDistance()
        implicit_surface.SetInput(surface)

        normal = get_surface_normals(surface, opening_center)
        normal = np.array(normal)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0.0, 0.0, -1.0])

        max_distance = -1
        best_point = None
        max_depth = reference_distance * 2.0
        coarse_samples = max(10, num_samples // 2)

        for i in range(1, coarse_samples + 1):
            depth = (max_depth / coarse_samples) * i
            test_point = np.array(opening_center) - normal * depth
            distance = implicit_surface.EvaluateFunction(test_point)
            if distance < 0 and abs(distance) > max_distance:
                max_distance = abs(distance)
                best_point = test_point.copy()

        if best_point is not None:
            best_depth = np.linalg.norm(best_point - np.array(opening_center))
            search_range = reference_distance * 0.3
            fine_samples = num_samples - coarse_samples

            for i in range(-fine_samples // 2, fine_samples // 2 + 1):
                if i == 0:
                    continue
                depth = best_depth + (search_range / max(1, fine_samples)) * i
                if depth <= 0:
                    continue
                test_point = np.array(opening_center) - normal * depth
                distance = implicit_surface.EvaluateFunction(test_point)
                if distance < 0 and abs(distance) > max_distance:
                    max_distance = abs(distance)
                    best_point = test_point.copy()

        if best_point is not None and implicit_surface.EvaluateFunction(best_point) < 0:
            return best_point.tolist()
        else:
            return None
    except Exception:
        return None


def find_center_by_reference_offset(surface, opening_center, reference_distance):
    try:
        normal = get_surface_normals(surface, opening_center)
        normal = np.array(normal)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0.0, 0.0, -1.0])

        implicit_surface = vtk.vtkImplicitPolyDataDistance()
        implicit_surface.SetInput(surface)

        candidate_in = np.array(opening_center) - normal * reference_distance
        candidate_out = np.array(opening_center) + normal * reference_distance

        if implicit_surface.EvaluateFunction(candidate_in) < 0:
            center_point = candidate_in.tolist()
        elif implicit_surface.EvaluateFunction(candidate_out) < 0:
            center_point = candidate_out.tolist()
        else:
            center_point = opening_center

        return center_point
    except Exception:
        return opening_center


def find_center_of_opening(surface, opening_center, opening_radius):
    """Encuentra el centro del lumen usando múltiples estrategias"""
    reference_distance = opening_radius

    # Estrategia 1: Método original de máxima distancia
    center_point = find_center_by_max_distance_with_reference(surface, opening_center, reference_distance)
    
    # Estrategia 2: Muestreo en grilla
    if center_point is None:
        center_point = find_center_by_grid_sampling(surface, opening_center, opening_radius)
    
    # Estrategia 3: Muestreo aleatorio
    if center_point is None:
        center_point = find_center_by_random_sampling(surface, opening_center, opening_radius)
    
    # Estrategia 4: Proyección desde superficie
    if center_point is None:
        center_point = find_center_by_surface_projection(surface, opening_center, opening_radius)
    
    # Estrategia 5: Método de referencia offset
    if center_point is None:
        center_point = find_center_by_reference_offset(surface, opening_center, reference_distance)
    
    # Estrategia 6: Método conservador (fallback)
    if center_point is None:
        normal = get_surface_normals(surface, opening_center)
        offset_distance = min(reference_distance * 0.5, 2.0)
        center_point = [
            opening_center[0] - normal[0] * offset_distance,
            opening_center[1] - normal[1] * offset_distance,
            opening_center[2] - normal[2] * offset_distance
        ]

    return center_point


def adjust_point_position(point, surface, offset_distance=0.5):
    try:
        normal = get_surface_normals(surface, point)
        adjusted_point = [
            point[0] - normal[0] * offset_distance,
            point[1] - normal[1] * offset_distance,
            point[2] - normal[2] * offset_distance
        ]
        return adjusted_point
    except:
        return point


# ===================== CHEQUEOS DE CONECTIVIDAD =====================

def test_connectivity_with_multiple_strategies(start_point, end_point, surface, max_attempts=5):
    """Intenta conectar dos puntos usando múltiples estrategias"""
    
    # Estrategia 1: Conexión directa
    if test_connectivity_single(start_point, end_point, surface):
        return True, end_point
    
    # Estrategia 2: Ajuste ligero del punto final
    adjusted_end = adjust_point_position(end_point, surface, 0.2)
    if test_connectivity_single(start_point, adjusted_end, surface):
        return True, adjusted_end
    
    # Estrategia 3: Ajuste moderado del punto final
    adjusted_end_2 = adjust_point_position(end_point, surface, 0.5)
    if test_connectivity_single(start_point, adjusted_end_2, surface):
        return True, adjusted_end_2
    
    # Estrategia 4: Proyección a superficie + ajuste
    projected_end = list(project_point_to_surface(end_point, surface))
    adjusted_projected = adjust_point_position(projected_end, surface, 0.3)
    if test_connectivity_single(start_point, adjusted_projected, surface):
        return True, adjusted_projected
    
    # Estrategia 5: Múltiples intentos con perturbación aleatoria
    for i in range(max_attempts):
        noise_scale = 0.3 * (i + 1)
        noise = np.random.normal(0, noise_scale, 3)
        perturbed_point = [end_point[j] + noise[j] for j in range(3)]
        adjusted_perturbed = adjust_point_position(perturbed_point, surface, 0.3)
        if test_connectivity_single(start_point, adjusted_perturbed, surface):
            return True, adjusted_perturbed
    
    # Estrategia 6: Intentar múltiples puntos a lo largo de la normal
    normal = get_surface_normals(surface, end_point)
    for depth_factor in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        for direction in [-1, 1]:
            test_point = [
                end_point[0] + normal[0] * depth_factor * direction,
                end_point[1] + normal[1] * depth_factor * direction,
                end_point[2] + normal[2] * depth_factor * direction
            ]
            if test_connectivity_single(start_point, test_point, surface):
                return True, test_point
    
    # Estrategia 7: Muestreo aleatorio agresivo
    for radius_factor in [0.5, 1.0, 1.5, 2.0]:
        for _ in range(20):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, radius_factor)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            random_point = [
                end_point[0] + x,
                end_point[1] + y,
                end_point[2] + z
            ]
            
            adjusted_random = adjust_point_position(random_point, surface, 0.2)
            if test_connectivity_single(start_point, adjusted_random, surface):
                return True, adjusted_random
    
    return False, end_point


def test_connectivity_single(start_point, end_point, surface):
    try:
        test_filter = vmtkscripts.vmtkCenterlines()
        test_filter.Surface = surface
        test_filter.SeedSelectorName = "pointlist"
        test_filter.SourcePoints = list(start_point)
        test_filter.TargetPoints = list(end_point)
        test_filter.AppendEndPoints = 0
        test_filter.CenterlineResampling = 0

        import sys, io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            test_filter.Execute()
            result_lines = test_filter.Centerlines.GetLines()
            success = result_lines.GetNumberOfCells() > 0
            if success:
                lines = test_filter.Centerlines.GetLines()
                lines.InitTraversal()
                idList = vtk.vtkIdList()
                lines.GetNextCell(idList)
                if idList.GetNumberOfIds() > 1:
                    last_point_id = idList.GetId(idList.GetNumberOfIds() - 1)
                    last_point = test_filter.Centerlines.GetPoint(last_point_id)
                    distance_to_target = np.linalg.norm(np.array(last_point) - np.array(end_point))
                    total_length = 0.0
                    for i in range(1, idList.GetNumberOfIds()):
                        p1 = test_filter.Centerlines.GetPoint(idList.GetId(i-1))
                        p2 = test_filter.Centerlines.GetPoint(idList.GetId(i))
                        dist = np.linalg.norm(np.array(p2) - np.array(p1))
                        total_length += dist
                    success = (distance_to_target < 1.0) and (total_length > 0.1)
        except:
            success = False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return success
    except:
        return False


# ===================== DETECCIÓN DE ABERTURAS =====================

def get_largest_opening_as_start(surface):
    connectivity = vtk.vtkFeatureEdges()
    connectivity.SetInputData(surface)
    connectivity.BoundaryEdgesOn()
    connectivity.FeatureEdgesOff()
    connectivity.NonManifoldEdgesOff()
    connectivity.ManifoldEdgesOff()
    connectivity.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputData(connectivity.GetOutput())
    stripper.Update()

    loops = stripper.GetOutput()
    n_loops = loops.GetNumberOfCells()

    max_radius = -1
    start_opening_info = None
    end_openings_info = []

    for i in range(n_loops):
        cell = loops.GetCell(i)
        points_ids = cell.GetPointIds()
        coords = np.array([loops.GetPoint(points_ids.GetId(j)) for j in range(points_ids.GetNumberOfIds())])

        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        radius = distances.mean()

        opening_info = {
            'index': i,
            'center': center,
            'radius': radius,
            'border_coords': coords
        }

        if radius > max_radius:
            if start_opening_info is not None:
                end_openings_info.append(start_opening_info)
            max_radius = radius
            start_opening_info = opening_info
        else:
            end_openings_info.append(opening_info)

    start_point = None
    if start_opening_info is not None:
        start_point = find_center_of_opening(
            surface,
            start_opening_info['center'].tolist(),
            start_opening_info['radius']
        )

        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(surface)
        safety_iter = 0
        while implicit.EvaluateFunction(start_point) >= 0 and safety_iter < 10:
            start_point = adjust_point_position(start_point, surface, offset_distance=0.3)
            safety_iter += 1

    # Conexión persistente de aberturas
    valid_end_points = []
    failed_connections = []
    
    for opening_info in end_openings_info:
        i = opening_info['index']
        center = opening_info['center'].tolist()
        radius = opening_info['radius']
        border_coords = opening_info['border_coords']

        connection_successful = False
        attempts = 0
        max_center_attempts = 3
        
        while not connection_successful and attempts < max_center_attempts:
            attempts += 1
            
            if attempts == 1:
                lumen_center = find_center_of_opening(surface, center, radius)
            elif attempts == 2:
                normal = get_surface_normals(surface, center)
                conservative_depth = radius * 0.3
                lumen_center = [
                    center[0] - normal[0] * conservative_depth,
                    center[1] - normal[1] * conservative_depth,
                    center[2] - normal[2] * conservative_depth
                ]
            else:
                normal = get_surface_normals(surface, center)
                aggressive_depth = radius * 1.5
                lumen_center = [
                    center[0] - normal[0] * aggressive_depth,
                    center[1] - normal[1] * aggressive_depth,
                    center[2] - normal[2] * aggressive_depth
                ]
            
            is_connected, final_coords = test_connectivity_with_multiple_strategies(
                start_point, lumen_center, surface, max_attempts=5
            )
            
            if is_connected:
                valid_end_points.append(final_coords)
                connection_successful = True
        
        if not connection_successful:
            failed_connections.append({
                'index': i,
                'center': center,
                'radius': radius,
                'last_attempted_lumen_center': lumen_center
            })

    # Intento final para aberturas fallidas
    if failed_connections:
        for failed_info in failed_connections:
            i = failed_info['index']
            center = failed_info['center']
            radius = failed_info['radius']
            
            # Estrategia desesperada: usar directamente el centro de la abertura
            is_connected, final_coords = test_connectivity_with_multiple_strategies(
                start_point, center, surface, max_attempts=10
            )
            
            if is_connected:
                valid_end_points.append(final_coords)
            else:
                # Última estrategia: múltiples puntos alrededor del borde
                border_coords = failed_info.get('border_coords', [])
                if len(border_coords) > 0:
                    border_sample_indices = np.linspace(0, len(border_coords)-1, min(5, len(border_coords)), dtype=int)
                    
                    for border_idx in border_sample_indices:
                        border_point = border_coords[border_idx].tolist()
                        adjusted_border = adjust_point_position(border_point, surface, 0.5)
                        
                        is_connected, final_coords = test_connectivity_with_multiple_strategies(
                            start_point, adjusted_border, surface, max_attempts=3
                        )
                        
                        if is_connected:
                            valid_end_points.append(final_coords)
                            break
    
    return start_point, valid_end_points


# ===================== RECORTE DE CENTERLINES =====================

def _compute_model_scale(surface):
    b = surface.GetBounds()
    dx, dy, dz = b[1]-b[0], b[3]-b[2], b[5]-b[4]
    diag = (dx*dx + dy*dy + dz*dz) ** 0.5
    return max(diag, 1.0)


def clip_centerlines_with_directional_verification(centerlines, surface, start_point, keep_longest_segment=True, tol=None):
    """Recorta las centerlines usando verificación direccional y de radio desde el startpoint"""
    if tol is None:
        tol = _compute_model_scale(surface) * 1e-4

    out_points = vtk.vtkPoints()
    out_lines = vtk.vtkCellArray()

    lines = centerlines.GetLines()
    lines.InitTraversal()
    idList = vtk.vtkIdList()

    for line_idx in range(lines.GetNumberOfCells()):
        lines.GetNextCell(idList)
        
        # Extraer puntos de esta centerline
        centerline_points = []
        for i in range(idList.GetNumberOfIds()):
            pid = idList.GetId(i)
            pt = centerlines.GetPoint(pid)
            centerline_points.append(pt)
        
        if len(centerline_points) < 2:
            continue
        
        # Aplicar recorte direccional y de radio
        trimmed_points = trim_centerline_from_start(
            centerline_points, start_point, surface
        )
        
        if len(trimmed_points) < 2:
            continue
        
        # Agregar puntos recortados a la salida
        new_ids = vtk.vtkIdList()
        for pt in trimmed_points:
            nid = out_points.InsertNextPoint(pt)
            new_ids.InsertNextId(nid)
        
        out_lines.InsertNextCell(new_ids)

    clipped = vtk.vtkPolyData()
    clipped.SetPoints(out_points)
    clipped.SetLines(out_lines)
    
    return clipped


def clip_centerlines_to_surface(centerlines, surface, keep_longest_segment=True, tol=None):
    """Recorta las centerlines eliminando cualquier tramo fuera del modelo"""
    if tol is None:
        tol = _compute_model_scale(surface) * 1e-4

    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(surface)

    def is_inside(pt):
        d = float(implicit.EvaluateFunction(pt))
        return d <= tol

    out_points = vtk.vtkPoints()
    out_lines = vtk.vtkCellArray()

    lines = centerlines.GetLines()
    lines.InitTraversal()
    idList = vtk.vtkIdList()

    for _ in range(lines.GetNumberOfCells()):
        lines.GetNextCell(idList)
        # Recolectar segmentos interiores contiguos
        segments = []
        current_segment = []

        for i in range(idList.GetNumberOfIds()):
            pid = idList.GetId(i)
            pt = centerlines.GetPoint(pid)
            if is_inside(pt):
                current_segment.append(pt)
            else:
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = []
        if len(current_segment) >= 2:
            segments.append(current_segment)

        if not segments:
            continue

        if keep_longest_segment:
            def seg_length(seg_pts):
                L = 0.0
                for k in range(1, len(seg_pts)):
                    p0 = np.array(seg_pts[k-1])
                    p1 = np.array(seg_pts[k])
                    L += float(np.linalg.norm(p1 - p0))
                return L
            segments.sort(key=seg_length, reverse=True)
            segments = [segments[0]]

        # Insertar segmentos como líneas independientes
        for seg in segments:
            new_ids = vtk.vtkIdList()
            for p in seg:
                nid = out_points.InsertNextPoint(p)
                new_ids.InsertNextId(nid)
            if new_ids.GetNumberOfIds() >= 2:
                out_lines.InsertNextCell(new_ids)

    clipped = vtk.vtkPolyData()
    clipped.SetPoints(out_points)
    clipped.SetLines(out_lines)
    return clipped


# ===================== PIPELINE PRINCIPAL =====================

def extract_centerlines_and_points(input_vtk_file, output_csv_file=None, output_vtk_file=None, verbose=False):
    """
    Extrae centerlines de un modelo VTK y guarda los datos
    
    Parámetros:
    - input_vtk_file: archivo .vtk de entrada
    - output_csv_file: archivo CSV de salida (opcional, None para no generar CSV)
    - output_vtk_file: archivo VTK de salida (opcional, None para no generar VTK)
    - verbose: si True, muestra información detallada
    """
    try:
        # Leer modelo
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_vtk_file)
        reader.Update()
        surface = reader.GetOutput()

        if verbose:
            print(f"Modelo cargado: {surface.GetNumberOfPoints()} puntos, {surface.GetNumberOfCells()} celdas")

        # Calcular start y endpoints centrados en el lumen
        start_point, end_points = get_largest_opening_as_start(surface)
        if not end_points:
            raise ValueError("No se encontraron puntos finales conectados al punto de inicio")

        source_points_flat = list(start_point)
        target_points_flat = [coord for pt in end_points for coord in pt]

        if verbose:
            print(f"Punto de inicio: {source_points_flat}")
            print(f"Puntos finales: {len(end_points)}")

        centerlineFilter = vmtkscripts.vmtkCenterlines()
        centerlineFilter.Surface = surface
        centerlineFilter.SeedSelectorName = "pointlist"
        centerlineFilter.SourcePoints = source_points_flat
        centerlineFilter.TargetPoints = target_points_flat
        centerlineFilter.AppendEndPoints = 0
        centerlineFilter.CenterlineResampling = 0
        centerlineFilter.RadiusArrayName = "MaximumInscribedSphereRadius"
        centerlineFilter.OutputLength = 0
        # --- Mejora: resampleo denso para modelos con curvas ---
        diag = _compute_model_scale(surface)
        centerlineFilter.CenterlineResampling = 1
        centerlineFilter.ResamplingStepLength = diag * 0.005
        centerlineFilter.AppendEndPoints = 1


        # Suprimir salida de VMTK
        import sys, io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if not verbose:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        
        try:
            centerlineFilter.Execute()
            raw_centerlines = centerlineFilter.Centerlines

            # Aplicar recorte direccional mejorado
            centerlines = clip_centerlines_with_directional_verification(
                raw_centerlines, surface, start_point, keep_longest_segment=True
            )
        finally:
            if not verbose:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Extraer puntos y longitudes del resultado recortado
        output_data = []
        lines = centerlines.GetLines()
        lines.InitTraversal()
        idList = vtk.vtkIdList()
        centerline_lengths = {}
        valid_centerlines = 0

        for line_id in range(lines.GetNumberOfCells()):
            lines.GetNextCell(idList)
            line_length = 0.0
            previous_point = None
            point_coords = []

            for i in range(idList.GetNumberOfIds()):
                point_id = idList.GetId(i)
                point = centerlines.GetPoint(point_id)
                point_coords.append(point)
                if previous_point is not None:
                    dx = point[0] - previous_point[0]
                    dy = point[1] - previous_point[1]
                    dz = point[2] - previous_point[2]
                    segment_length = (dx**2 + dy**2 + dz**2)**0.5
                    line_length += segment_length
                previous_point = point

            centerline_lengths[line_id] = line_length
            if line_length > 0.1:
                for point in point_coords:
                    output_data.append([line_id, point[0], point[1], point[2], line_length])
                valid_centerlines += 1

        # Guardar CSV solo si se especifica un archivo de salida
        if output_csv_file and output_data:
            np.savetxt(output_csv_file,
                       np.array(output_data),
                       delimiter=",",
                       header="CenterlineID,X,Y,Z,Length",
                       comments="",
                       fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])
            if verbose:
                print(f"Datos CSV guardados en {output_csv_file}")
        elif output_csv_file and not output_data:
            if verbose:
                print("No hay datos válidos para guardar en CSV")

        # Guardar VTK si se especifica
        if output_vtk_file:
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(output_vtk_file)
            writer.SetInputData(centerlines)
            writer.Write()
            if verbose:
                print(f"Datos VTK guardados en {output_vtk_file}")

        return centerlines, len(output_data), valid_centerlines, centerline_lengths

    except Exception as e:
        if verbose:
            print(f"Error durante la ejecución: {str(e)}")
        raise e


# ===================== PROCESAMIENTO BATCH =====================

def process_batch_centerlines(input_folder, output_folder, file_pattern="*_model.vtk", 
                            generate_csv=True, generate_vtk=True, verbose=False):
    """
    Procesa múltiples archivos VTK en lote para extraer centerlines
    
    Parámetros:
    - input_folder: carpeta con los archivos VTK de entrada
    - output_folder: carpeta donde guardar los resultados
    - file_pattern: patrón de archivos a procesar (por defecto "*_model.vtk")
    - generate_csv: si True, genera archivos CSV (por defecto True)
    - generate_vtk: si True, genera archivos VTK (por defecto True)
    - verbose: si True, muestra información detallada del procesamiento
    """
    
    # Crear carpeta de salida si no existe
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Buscar archivos que coincidan con el patrón
    input_path = Path(input_folder)
    vtk_files = list(input_path.glob(file_pattern))
    
    if not vtk_files:
        print(f"No se encontraron archivos con patrón '{file_pattern}' en {input_folder}")
        return
    
    print(f"Encontrados {len(vtk_files)} archivos para procesar")
    if generate_csv and generate_vtk:
        print("Generando archivos CSV y VTK")
    elif generate_csv:
        print("Generando solo archivos CSV")
    elif generate_vtk:
        print("Generando solo archivos VTK")
    else:
        print("ADVERTENCIA: No se generarán archivos de salida (generate_csv=False, generate_vtk=False)")
    
    results_summary = []
    
    for i, vtk_file in enumerate(vtk_files):
        # Extraer nombre base del archivo (sin extensión)
        file_stem = vtk_file.stem
        # Remover "_model" si está presente
        if file_stem.endswith("_model"):
            base_name = file_stem[:-6]  # Remover "_model"
        else:
            base_name = file_stem
        
        # Definir archivos de salida según las opciones
        output_csv = Path(output_folder) / f"{base_name}_centerlines.csv" if generate_csv else None
        output_vtk = Path(output_folder) / f"{base_name}_centerlines.vtk" if generate_vtk else None
        
        print(f"\n[{i+1}/{len(vtk_files)}] Procesando: {vtk_file.name}")
        
        try:
            centerlines, num_points, num_valid_centerlines, lengths = extract_centerlines_and_points(
                str(vtk_file), 
                str(output_csv) if output_csv else None,
                str(output_vtk) if output_vtk else None,
                verbose=verbose
            )
            
            result_info = {
                'file': vtk_file.name,
                'status': 'SUCCESS',
                'points': num_points,
                'centerlines': num_valid_centerlines
            }
            
            if generate_csv:
                result_info['csv_output'] = output_csv.name if output_csv else None
            if generate_vtk:
                result_info['vtk_output'] = output_vtk.name if output_vtk else None
                
            results_summary.append(result_info)
            
            output_info = []
            if generate_csv and num_points > 0:
                output_info.append("CSV")
            if generate_vtk:
                output_info.append("VTK")
            
            output_str = f" ({', '.join(output_info)} generados)" if output_info else ""
            
            if verbose:
                print(f"  ✓ Completado: {num_points} puntos, {num_valid_centerlines} centerlines{output_str}")
            else:
                print(f"  ✓ Completado: {num_points} puntos, {num_valid_centerlines} centerlines{output_str}")
            
        except Exception as e:
            results_summary.append({
                'file': vtk_file.name,
                'status': 'ERROR',
                'error': str(e),
                'points': 0,
                'centerlines': 0
            })
            
            print(f"  ✗ Error: {str(e)}")
    
    # Mostrar resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DEL PROCESAMIENTO BATCH")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    failed = len(results_summary) - successful
    
    print(f"Archivos procesados exitosamente: {successful}")
    print(f"Archivos con errores: {failed}")
    print(f"Total de archivos: {len(results_summary)}")
    
    if successful > 0:
        total_points = sum(r['points'] for r in results_summary if r['status'] == 'SUCCESS')
        total_centerlines = sum(r['centerlines'] for r in results_summary if r['status'] == 'SUCCESS')
        print(f"Total de puntos extraídos: {total_points}")
        print(f"Total de centerlines válidas: {total_centerlines}")
        
        if generate_csv:
            csv_generated = sum(1 for r in results_summary 
                              if r['status'] == 'SUCCESS' and r.get('points', 0) > 0)
            print(f"Archivos CSV generados: {csv_generated}")
        
        if generate_vtk:
            vtk_generated = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
            print(f"Archivos VTK generados: {vtk_generated}")
    
    if failed > 0:
        print(f"\nArchivos con errores:")
        for result in results_summary:
            if result['status'] == 'ERROR':
                print(f"  - {result['file']}: {result['error']}")
    
    print(f"\nArchivos de salida guardados en: {output_folder}")
    
    return results_summary


# ===================== FUNCIONES DE CONVENIENCIA =====================

def process_single_file(input_vtk_file, output_folder=None, base_name=None, 
                       generate_csv=True, generate_vtk=True, verbose=False):
    """
    Procesa un solo archivo VTK
    
    Parámetros:
    - input_vtk_file: ruta del archivo VTK de entrada
    - output_folder: carpeta de salida (por defecto: misma carpeta del archivo de entrada)
    - base_name: nombre base para archivos de salida (por defecto: nombre del archivo sin extensión)
    - generate_csv: si True, genera archivo CSV
    - generate_vtk: si True, genera archivo VTK
    - verbose: si True, muestra información detallada
    """
    input_path = Path(input_vtk_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"El archivo {input_vtk_file} no existe")
    
    # Determinar carpeta de salida
    if output_folder is None:
        output_folder = input_path.parent
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determinar nombre base
    if base_name is None:
        file_stem = input_path.stem
        if file_stem.endswith("_model"):
            base_name = file_stem[:-6]
        else:
            base_name = file_stem
    
    # Definir archivos de salida
    output_csv = output_folder / f"{base_name}_centerlines.csv" if generate_csv else None
    output_vtk_out = output_folder / f"{base_name}_centerlines.vtk" if generate_vtk else None
    
    print(f"Procesando archivo: {input_path.name}")
    
    try:
        centerlines, num_points, num_valid_centerlines, lengths = extract_centerlines_and_points(
            str(input_path),
            str(output_csv) if output_csv else None,
            str(output_vtk_out) if output_vtk_out else None,
            verbose=verbose
        )
        
        output_info = []
        if generate_csv and num_points > 0:
            output_info.append(f"CSV: {output_csv.name}")
        if generate_vtk:
            output_info.append(f"VTK: {output_vtk_out.name}")
        
        if output_info:
            print(f"✓ Completado: {num_points} puntos, {num_valid_centerlines} centerlines")
            print(f"  Archivos generados: {', '.join(output_info)}")
        else:
            print(f"✓ Procesamiento completado: {num_points} puntos, {num_valid_centerlines} centerlines")
        
        return centerlines, num_points, num_valid_centerlines, lengths
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise


# ===================== FUNCIÓN PRINCIPAL =====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Uso del script:")
        print("  python script.py <carpeta_entrada> <carpeta_salida> [patron] [csv] [vtk] [verbose]")
        print()
        print("Parámetros:")
        print("  carpeta_entrada: carpeta con archivos VTK")
        print("  carpeta_salida:  carpeta donde guardar resultados") 
        print("  patron:          patrón de archivos (opcional, por defecto '*_model.vtk')")
        print("  csv:             generar CSV - True/False (opcional, por defecto True)")
        print("  vtk:             generar VTK - True/False (opcional, por defecto True)")
        print("  verbose:         información detallada - True/False (opcional, por defecto False)")
        print()
        print("Ejemplos:")
        print("  python script.py ./models ./output")
        print("  python script.py ./models ./output *.vtk True False True")
        print("  python script.py ./models ./output *_model.vtk False True False")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    file_pattern = sys.argv[3] if len(sys.argv) > 3 else "*_model.vtk"
    generate_csv = sys.argv[4].lower() != 'false' if len(sys.argv) > 4 else True
    generate_vtk = sys.argv[5].lower() != 'false' if len(sys.argv) > 5 else True
    verbose = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    
    if not os.path.exists(input_folder):
        print(f"Error: La carpeta de entrada '{input_folder}' no existe")
        sys.exit(1)
    
    process_batch_centerlines(input_folder, output_folder, file_pattern, 
                            generate_csv, generate_vtk, verbose)
