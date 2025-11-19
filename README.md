# Trabajo de Grado: Electrodos Virtuales en TIE 🔬⚡

<div align="center">

<!-- Logo Universidad del Cauca -->
<!-- Reemplaza 'logo-unicauca.png' con el nombre de tu logo -->
<img src="images/logo-unicauca.png" alt="Universidad del Cauca" width="200"/>

---

![EIT Banner](https://via.placeholder.com/800x200/667eea/ffffff?text=Tomograf%C3%ADa+por+Impedancia+El%C3%A9ctrica)

### **Comparación entre Métodos de Electrodos Virtuales**
*Efecto en la Resolución Espacial e Inmunidad al Ruido*

---

🏛️ **Universidad del Cauca** | 📍 Popayán, Colombia | 📅 2025

**Facultad de Ingeniería en Electrónica y Telecomunicaciones**  
**Programa de Ingeniería en Automática Industrial**

---

</div>

## 🎯 ¿Qué es este proyecto?

Este trabajo de grado compara **6 métodos diferentes** para generar **Electrodos Virtuales** en sistemas de Tomografía por Impedancia Eléctrica (TIE), logrando mejorar la **resolución espacial** de las imágenes sin necesidad de hardware adicional.

> **Objetivo:** Transformar un sistema de **8 electrodos** para que funcione como si tuviera **16 electrodos**, usando algoritmos inteligentes 🧠

---

## 📊 Métodos Evaluados

<table>
<tr>
<td width="50%">

### 🔢 Métodos Clásicos
- ✅ **Interpolación Lineal**
- ✅ **Interpolación Cúbica** 
- ✅ **Interpolación por Splines (PCHIP)**

</td>
<td width="50%">

### 🤖 Métodos Avanzados
- ✅ **Método α + Algoritmo Genético**
- ✅ **CNN de Aumento de Datos**
- ✅ **Modelo Híbrido Físico+NN**

</td>
</tr>
</table>

---

## 🗂️ Estructura del Repositorio

```
📦 Trabajo_de_grado_TIE/
│
├── 📄 README.md
│
├── 🔢 MÉTODOS DE INTERPOLACIÓN
│   ├── Inter_lineal.m
│   ├── inter_cubic.m
│   ├── inter_spline.m
│   └── Comparacion_Metodos_Interpolacion_lineal_cu...
│
├── 🧬 MÉTODO ALPHA CON GA
│   ├── optimizacion_alpha_GA.m
│   ├── custom_fitness.m
│   ├── Comparacion_valores_diff_alpha.m
│   └── Comparacion_alpha_fijo_16FEM_vs_8FEM_8EV.m
│
├── 🖼️ RECONSTRUCCIÓN DE IMÁGENES
│   ├── Reconstruccion_inter_lineal.m
│   ├── Reconstruccion_alpha_fijo.m
│   └── Reconstruccion_de_alpha_dinámico.m
│
├── 🤖 MÉTODOS BASADOS EN ML/DL
│   ├── main_hybrid_residual_v3.m
│   └── Metodo de Aumento de Datos.rar
│
├── 📂 images/
│   ├── logo-unicauca.png
│   └── logo-grupo-automatica.png
│
└── 📚 DOCUMENTACIÓN
    └── [Documento completo del trabajo]
```

---

## 🛠️ Tecnologías Utilizadas

<div align="center">

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-orange?style=for-the-badge&logo=mathworks)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![EIDORS](https://img.shields.io/badge/EIDORS-3.12-green?style=for-the-badge)

</div>

### Herramientas Clave:
- 🔷 **MATLAB + EIDORS** para simulación FEM
- 🧠 **TensorFlow/Keras** para redes neuronales
- 🧬 **Global Optimization Toolbox** para algoritmos genéticos
- 📊 **5 Algoritmos de Reconstrucción:** Tikhonov, NOSER, Laplaciano, Bayesiano, Variación Total

---

## 📈 Resultados Destacados

<div align="center">

| Método | CC ↑ | ER ↓ | Destacado |
|--------|------|------|-----------|
| **Interpolación Clásica** | 0.75-0.82 | 0.18-0.25 | ❌ Limitaciones fundamentales |
| **Método α-GA** | 0.85-0.87 | 0.16-0.18 | ⚠️ Bueno pero costoso |
| **CNN Aumento Datos** | 0.91-0.94 | 0.12-0.15 | ✅ Excelente concordancia |
| **Modelo Híbrido** | 0.94-0.96 | 0.09-0.12 | 🏆 **43% reducción error** |

</div>

### 🎯 Conclusión Principal

Los métodos basados en **Deep Learning** (CNN y modelo híbrido) superan significativamente a los enfoques clásicos, logrando imágenes de mayor calidad con mayor robustez al ruido.

---

## 🚀 Cómo Usar Este Repositorio

### 1️⃣ Clonar el Repositorio
```bash
git clone https://github.com/JulianBurgos03/Trabajo_de_grado_TIE.git
cd Trabajo_de_grado_TIE
```

### 2️⃣ Configurar MATLAB
```matlab
% Agregar EIDORS al path
addpath(genpath('ruta/a/eidors'));

% Ejecutar métodos de interpolación
run Inter_lineal.m
run inter_cubic.m
run inter_spline.m
```

### 3️⃣ Optimización con GA
```matlab
% Optimizar parámetro alpha con algoritmo genético
run optimizacion_alpha_GA.m
```

### 4️⃣ Reconstruir Imágenes
```matlab
% Reconstrucción con diferentes métodos
run Reconstruccion_inter_lineal.m
run Reconstruccion_de_alpha_dinámico.m
```

---

## 📚 Citar Este Trabajo

Si usas este código o metodología, por favor cita:

```bibtex
@mastersthesis{BurgosFernandez2025TIE,
  author = {Burgos Ayala, Ángel Julián and Fernández Pomeo, Juan José},
  title  = {Comparación entre Métodos de Electrodos Virtuales en TIE},
  school = {Universidad del Cauca},
  year   = {2025},
  address = {Popayán, Colombia}
}
```

---

## 👥 Autores

<div align="center">

<table>
<tr>
<td align="center" width="33%">
<img src="https://avatars.githubusercontent.com/u/placeholder?v=4" width="100px;" alt="Ángel Julián Burgos Ayala"/><br>
<b>Ángel Julián Burgos Ayala</b><br>
<i>Ingeniería en Automática Industrial</i><br>
Universidad del Cauca<br><br>
<a href="https://www.linkedin.com/in/angel-burgos-ingaut/">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
<a href="https://www.researchgate.net/profile/Angel-Burgos-3?ev=hdr_xprf">
  <img src="https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white"/>
</a><br>
📧 <a href="mailto:ajburgos@unicauca.edu.co">ajburgos@unicauca.edu.co</a>
</td>

<td align="center" width="33%">
<img src="https://avatars.githubusercontent.com/u/placeholder?v=4" width="100px;" alt="Juan José Fernández Pomeo"/><br>
<b>Juan José Fernández Pomeo</b><br>
<i>Ingeniería en Automática Industrial</i><br>
Universidad del Cauca<br><br>
<a href="https://www.linkedin.com/in/juan-jos%C3%A9-fern%C3%A1ndez-pomeo-74830b2a9/">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
<a href="#">
  <img src="https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white"/>
</a><br>
📧 <a href="mailto:jujofernandez@unicauca.edu.co">jujofernandez@unicauca.edu.co</a>
</td>

<td align="center" width="33%">
<img src="https://avatars.githubusercontent.com/u/placeholder?v=4" width="100px;" alt="Víctor Hugo Mosquera Leyton"/><br>
<b>Ph.D. Víctor Hugo Mosquera Leyton</b><br>
<i>Director del Trabajo</i><br>
Ciencias de la Electrónica<br>
Universidad del Cauca<br>
<a href="https://www.linkedin.com/in/v%C3%ADctor-hugo-mosquera-a7436833/">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
<a href="https://www.researchgate.net/profile/Victor-Mosquera-2">
  <img src="https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white"/>
</a><br>
📧 <a href="mailto:mosquera@unicauca.edu.co">mosquera@unicauca.edu.co</a>
</td>
</tr>
</table>

</div>

---

## 📞 Contacto

<div align="center">

### 📧 Correos Electrónicos

**Ángel Julián Burgos Ayala:** [ajburgos@unicauca.edu.co](mailto:ajburgos@unicauca.edu.co)  
**Juan José Fernández Pomeo:** [jujofernandez@unicauca.edu.co](mailto:jujofernandez@unicauca.edu.co)  
**Ph.D. Víctor Hugo Mosquera Leyton:** [mosquera@unicauca.edu.co](mailto:mosquera@unicauca.edu.co)

---

### 🏛️ Universidad del Cauca
Facultad de Ingeniería en Electrónica y Telecomunicaciones  
Calle 5 No. 4-70, Popayán, Cauca, Colombia

---

### 🌐 Redes Académicas

[![GitHub](https://img.shields.io/badge/GitHub-JulianBurgos03-181717?style=for-the-badge&logo=github)](https://github.com/JulianBurgos03)
[![Universidad](https://img.shields.io/badge/Web-Universidad_del_Cauca-blue?style=for-the-badge)](https://www.unicauca.edu.co)

</div>

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

**Uso académico:** ✅ Libre con atribución apropiada  
**Uso comercial:** ⚠️ Contactar a los autores

---

## 🙏 Agradecimientos

Agradecimiento especial al **Ph.D. Víctor Hugo Mosquera Leyton** por su guía y apoyo durante el desarrollo de esta investigación, y a la **Universidad del Cauca** por proporcionar los recursos e instalaciones necesarios para la realización de este trabajo.

Agradecemos también al **Grupo de Investigación en Automática Industrial** por el respaldo institucional y académico brindado durante todo el proceso.

---

<div align="center">

### 🎓 Universidad del Cauca | 2025

**Grupo de Investigación en Automática Industrial**

<!-- Logo del Grupo de Investigación en Automática -->
<!-- Reemplaza 'logo-grupo-automatica.png' con el nombre de tu logo -->
<img src="images/logo-grupo-automatica.png" alt="Grupo de Investigación en Automática" width="150"/>

---

**Desarrollado con dedicación para avanzar en imagenología médica no invasiva** 💙

![Footer](https://via.placeholder.com/800x100/1e3c72/ffffff?text=Tomograf%C3%ADa+por+Impedancia+El%C3%A9ctrica+-+Universidad+del+Cauca)

---

**⭐ Si este proyecto te fue útil, considera darle una estrella!**

*Made with ❤️ for advancing medical imaging technologies*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-orange.svg)](https://www.mathworks.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

</div>
